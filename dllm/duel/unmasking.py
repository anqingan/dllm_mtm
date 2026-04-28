"""Deterministic unmasking rules for DUEL-guided decoding.

Every rule is fully deterministic: given the same (input_ids, logits,
candidate_mask, k) it always returns the same positions.  Tie-breaking
uses the leftmost index.

Supported rules:
  - left_to_right:  k leftmost masked positions
  - greedy_confidence: k positions with highest top-1 softmax probability
  - probability_margin: k positions with largest top-1/top-2 probability gap
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def select_positions(
    input_ids: torch.Tensor,  # [B, T]  current state (may contain mask tokens)
    logits: torch.Tensor,  # [B, T, V] model output logits
    candidate_mask: torch.Tensor,  # [B, T] bool – positions allowed to reveal
    rule: str,
    k: int,
    mask_token_id: int,
) -> list[list[int]]:
    """Select up to *k* positions to unmask per sample.

    Returns a list (length B) of lists of position indices.  Each inner list
    contains at most *k* indices.  If no candidates exist for a sample, its
    list is empty.
    """
    B, T = input_ids.shape
    results: list[list[int]] = []

    if k <= 0:
        return [[] for _ in range(B)]

    allowed_mask = candidate_mask & (input_ids == mask_token_id)

    for b in range(B):
        cand_idx = torch.where(allowed_mask[b])[0]  # 1-D tensor of valid indices
        if cand_idx.numel() == 0:
            results.append([])
            continue

        actual_k = min(k, cand_idx.numel())

        if rule == "left_to_right":
            # cand_idx is already sorted ascending
            selected = cand_idx[:actual_k].tolist()

        elif rule == "greedy_confidence":
            probs = F.softmax(logits[b], dim=-1)  # [T, V]
            top1_prob = probs.max(dim=-1).values  # [T]
            scores = top1_prob[cand_idx]
            # Stable sort: highest score first; ties broken by index (ascending)
            _, sort_order = torch.sort(scores, descending=True, stable=True)
            selected = cand_idx[sort_order[:actual_k]].tolist()

        elif rule == "probability_margin":
            probs = F.softmax(logits[b], dim=-1)  # [T, V]
            sorted_p, _ = torch.sort(probs, dim=-1, descending=True)  # [T, V]
            margin = sorted_p[:, 0] - sorted_p[:, 1]  # [T]
            scores = margin[cand_idx]
            _, sort_order = torch.sort(scores, descending=True, stable=True)
            selected = cand_idx[sort_order[:actual_k]].tolist()

        else:
            raise ValueError(f"Unknown unmask_rule: {rule!r}")

        results.append(selected)

    return results
