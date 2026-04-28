"""DUEL sampler: generate tokens in a masked region using a deterministic
unmasking rule for position selection, with stochastic token sampling.

Unmask positions are deterministic (same rule F as the scorer).
Token values are sampled from the model distribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dllm.duel.types import DuelProposalResult
from dllm.duel.unmasking import select_positions


def _proposal_log_probs(
    logits_1d: torch.Tensor,  # [V]
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
) -> torch.Tensor:
    """Return normalized proposal log-probs after temperature/support filters."""
    if temperature <= 0.0:
        raise ValueError("proposal temperature must be positive for exact log q")

    logits_1d = logits_1d / temperature
    if mask_token_id is not None:
        logits_1d = logits_1d.clone()
        logits_1d[mask_token_id] = -torch.inf

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits_1d.numel())
        values, _ = torch.topk(logits_1d, top_k)
        logits_1d = logits_1d.masked_fill(logits_1d < values[-1], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs > top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        logits_1d = logits_1d.masked_fill(
            torch.zeros_like(logits_1d, dtype=torch.bool).scatter(
                0, sorted_idx, mask
            ),
            float("-inf"),
        )

    if torch.isneginf(logits_1d).all():
        raise ValueError("proposal distribution has empty support")
    return F.log_softmax(logits_1d, dim=-1)


def _sample_token_from_logits(
    logits_1d: torch.Tensor,  # [V]
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a single token and return (token, proposal log-prob)."""
    log_probs = _proposal_log_probs(
        logits_1d,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        mask_token_id=mask_token_id,
    )
    probs = torch.exp(log_probs)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    token = token.squeeze()
    return token, log_probs[token]


@torch.inference_mode()
def duel_generate_region(
    model: torch.nn.Module,
    initial_state: torch.Tensor,  # [1, T]
    generation_mask: torch.Tensor,  # [1, T] bool – only these positions may change
    unmask_rule: str,
    positions_per_step: int,
    mask_token_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_trace: bool = False,
) -> DuelProposalResult:
    """Generate tokens in *generation_mask* region via deterministic unmasking.

    Returns (updated_state, trace_or_None).
    Positions outside generation_mask are never modified.
    """
    z = initial_state.clone()
    B, T = z.shape
    assert B == 1, "DUEL sampler currently requires batch_size=1"

    active = generation_mask.clone()
    trace_steps = [] if return_trace else None
    total_log_q = torch.tensor(0.0, device=z.device)
    per_step_logprobs: list[float] = []

    while True:
        remaining = active[0] & (z[0] == mask_token_id)
        if not remaining.any():
            break

        fwd_kwargs: dict = {}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids
        logits = model(z, **fwd_kwargs).logits  # [1, T, V]

        positions_per_sample = select_positions(
            input_ids=z,
            logits=logits,
            candidate_mask=active & (z == mask_token_id),
            rule=unmask_rule,
            k=positions_per_step,
            mask_token_id=mask_token_id,
        )

        if not positions_per_sample[0]:
            break

        for pos in positions_per_sample[0]:
            position_logits = logits[0, pos].clone()
            position_logits[mask_token_id] = -torch.inf
            sampled, log_q = _sample_token_from_logits(
                position_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
                generator=generator,
            )
            total_log_q = total_log_q + log_q
            per_step_logprobs.append(float(log_q.item()))
            z[0, pos] = sampled
            active[0, pos] = False

        if return_trace and trace_steps is not None:
            trace_steps.append({
                "positions": positions_per_sample[0],
                "tokens": [z[0, p].item() for p in positions_per_sample[0]],
            })

    return DuelProposalResult(
        sequence=z,
        trace={"steps": trace_steps} if return_trace else None,
        log_q=total_log_q,
        per_step_logprobs=per_step_logprobs,
    )


@torch.inference_mode()
def compute_duel_proposal_logprob(
    model: torch.nn.Module,
    initial_state: torch.Tensor,  # [1, T]
    target_state: torch.Tensor,  # [1, T]
    generation_mask: torch.Tensor,  # [1, T] bool
    unmask_rule: str,
    positions_per_step: int,
    mask_token_id: int,
    proposal_temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teacher-forced log q_Tprop(target_state | initial_state)."""
    z = initial_state.clone()
    B, _ = z.shape
    assert B == 1, "DUEL proposal replay currently requires batch_size=1"

    active = generation_mask.clone()
    total_log_q = torch.tensor(0.0, device=z.device)

    while True:
        remaining = active[0] & (z[0] == mask_token_id)
        if not remaining.any():
            break

        fwd_kwargs: dict = {}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids
        logits = model(z, **fwd_kwargs).logits

        positions_per_sample = select_positions(
            input_ids=z,
            logits=logits,
            candidate_mask=active & (z == mask_token_id),
            rule=unmask_rule,
            k=positions_per_step,
            mask_token_id=mask_token_id,
        )

        if not positions_per_sample[0]:
            break

        for pos in positions_per_sample[0]:
            token = target_state[0, pos]
            if token.item() == mask_token_id:
                raise ValueError(
                    "target_state contains mask_token_id in proposal replay region"
                )
            log_probs = _proposal_log_probs(
                logits[0, pos].clone(),
                temperature=proposal_temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
            )
            total_log_q = total_log_q + log_probs[token]
            z[0, pos] = token
            active[0, pos] = False

    return total_log_q
