"""DUEL sampler: generate tokens in a masked region using a deterministic
unmasking rule for position selection, with stochastic token sampling.

Unmask positions are deterministic (same rule F as the scorer).
Token values are sampled from the model distribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dllm.duel.unmasking import select_positions


def _sample_token_from_logits(
    logits_1d: torch.Tensor,  # [V]
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a single token from a 1-D logit vector."""
    if temperature != 0.0:
        logits_1d = logits_1d / temperature

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

    probs = F.softmax(logits_1d, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return token.squeeze()


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
) -> tuple[torch.Tensor, dict | None]:
    """Generate tokens in *generation_mask* region via deterministic unmasking.

    Returns (updated_state, trace_or_None).
    Positions outside generation_mask are never modified.
    """
    z = initial_state.clone()
    B, T = z.shape
    assert B == 1, "DUEL sampler currently requires batch_size=1"

    active = generation_mask.clone()
    trace_steps = [] if return_trace else None

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
            sampled = _sample_token_from_logits(
                position_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            z[0, pos] = sampled
            active[0, pos] = False

        if return_trace and trace_steps is not None:
            trace_steps.append({
                "positions": positions_per_sample[0],
                "tokens": [z[0, p].item() for p in positions_per_sample[0]],
            })

    return z, {"steps": trace_steps} if return_trace else None
