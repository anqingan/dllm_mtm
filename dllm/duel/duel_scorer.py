"""DUEL conditional log-likelihood scorer.

Computes the exact conditional likelihood of candidate tokens under a
deterministic unmasking rule F.  Scoring never samples tokens — every
revealed token comes from *target_tokens*.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from dllm.duel.unmasking import select_positions


@torch.inference_mode()
def compute_duel_conditional_loglikelihood(
    model: torch.nn.Module,
    initial_state: torch.Tensor,  # [1, T] partial state with masks
    target_tokens: torch.Tensor,  # [1, T] full candidate tokens
    score_mask: torch.Tensor,  # [1, T] bool – positions to accumulate
    candidate_mask: torch.Tensor,  # [1, T] bool – positions allowed to reveal
    unmask_rule: str,
    positions_per_step: int,
    mask_token_id: int,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    """Score *target_tokens* under DUEL conditional likelihood.

    Returns (total_log_likelihood_scalar, diagnostics_dict_or_None).
    total_log_likelihood has shape [] (scalar).
    """
    z = initial_state.clone()
    B, T = z.shape
    assert B == 1, "DUEL scorer currently requires batch_size=1"

    # Validate: target must not have mask tokens at positions we will reveal
    will_reveal = candidate_mask & (z == mask_token_id)
    offending = will_reveal & (target_tokens == mask_token_id)
    if offending.any():
        positions = torch.where(offending[0])[0].tolist()
        raise ValueError(
            f"target_tokens still contains mask_token_id at positions "
            f"{positions} that are scheduled for reveal"
        )

    total_ll = torch.tensor(0.0, device=z.device)
    active_candidates = candidate_mask.clone()
    step_diagnostics = [] if return_diagnostics else None

    while True:
        # Check remaining masked candidates
        remaining = active_candidates[0] & (z[0] == mask_token_id)
        if not remaining.any():
            break

        # Forward pass
        fwd_kwargs: dict = {}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids
        logits = model(z, **fwd_kwargs).logits  # [1, T, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [1, T, V]

        # Select positions deterministically
        positions_per_sample = select_positions(
            input_ids=z,
            logits=logits,
            candidate_mask=active_candidates & (z == mask_token_id),
            rule=unmask_rule,
            k=positions_per_step,
            mask_token_id=mask_token_id,
        )

        if not positions_per_sample[0]:
            break

        step_ll = 0.0
        for pos in positions_per_sample[0]:
            token = target_tokens[0, pos].item()
            lp = log_probs[0, pos, token].item()
            if score_mask[0, pos]:
                total_ll += lp
                step_ll += lp
            z[0, pos] = token
            active_candidates[0, pos] = False

        if return_diagnostics and step_diagnostics is not None:
            step_diagnostics.append({
                "positions": positions_per_sample[0],
                "step_ll": step_ll,
            })

    diag = None
    if return_diagnostics and step_diagnostics is not None:
        diag = {"steps": step_diagnostics}

    return total_ll, diag
