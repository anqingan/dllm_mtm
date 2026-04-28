import itertools
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# ---------------------------------------------------------------------------
# New unmasking-strategy helpers
# ---------------------------------------------------------------------------

def compute_confidence_scores(
    logits: torch.Tensor,
    x0: torch.Tensor,
    remasking: str,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute per-position confidence scores and (optionally) the full softmax
    distribution for downstream use (e.g. KLASS KL-divergence).

    Args:
        logits: [B, L, V] raw model outputs.
        x0: [B, L] predicted token ids (argmax of logits + optional noise).
        remasking: Strategy name.

    Returns:
        confidence: [B, L] scalar confidence per position.
        current_probs: [B, L, V] softmax probabilities or None (for "random").
    """
    if remasking in ("low_confidence", "greedy_confidence"):
        p = F.softmax(logits, dim=-1)
        confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        return confidence, p

    if remasking == "random":
        confidence = torch.rand(
            (logits.shape[0], logits.shape[1]), device=logits.device
        )
        return confidence, None

    if remasking == "probability_margin":
        p = F.softmax(logits, dim=-1)
        sorted_p, _ = torch.sort(p, dim=-1, descending=True)
        confidence = sorted_p[:, :, 0] - sorted_p[:, :, 1]
        return confidence, p

    if remasking == "entropy":
        p = F.softmax(logits, dim=-1)
        eps = 1e-10
        log_p = torch.log(p + eps)
        confidence = torch.sum(p * log_p, dim=-1)
        return confidence, p

    # For strategies that don't define their own confidence metric,
    # fall back to greedy confidence (probability of predicted token).
    if remasking in ("left_to_right", "confidence_threshold", "klass", "oracle"):
        p = F.softmax(logits, dim=-1)
        confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        return confidence, p

    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")


def select_transfer_positions(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    remasking: str,
    threshold: float | None = None,
    kl_threshold: float | None = None,
    kl_divergence: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Decide which masked positions to unmask this step.

    Args:
        confidence: [B, L] confidence scores (already masked-out non-mask positions
                    should be set to -inf by the caller, or we do it here).
        mask_index: [B, L] bool – True for currently masked positions.
        num_transfer_tokens: [B] int – target k per sample.
        remasking: Strategy name.
        threshold: Confidence threshold for adaptive strategies.
        kl_threshold: KL-divergence ceiling for KLASS.
        kl_divergence: [B, L] pre-computed KL(prev || curr), or None.

    Returns:
        transfer: [B, L] bool – True for positions to unmask.
    """
    B, L = confidence.shape
    device = confidence.device

    # Mask-filter: only masked positions are candidates
    neg_inf = torch.full_like(confidence, -float("inf"))
    filtered_conf = torch.where(mask_index, confidence, neg_inf)

    # --- Left-to-Right: select k smallest *indices* among masked positions ---
    if remasking == "left_to_right":
        transfer = torch.zeros(B, L, dtype=torch.bool, device=device)
        for j in range(B):
            k = int(num_transfer_tokens[j].item())
            if k <= 0:
                continue
            masked_positions = torch.where(mask_index[j])[0]
            if masked_positions.numel() == 0:
                continue
            k = min(k, masked_positions.numel())
            # smallest indices first (already sorted ascending)
            transfer[j, masked_positions[:k]] = True
        return transfer

    # --- Confidence Threshold: adaptive k, all positions >= threshold ---
    if remasking == "confidence_threshold":
        _threshold = threshold if threshold is not None else 0.95
        transfer = torch.zeros(B, L, dtype=torch.bool, device=device)
        for j in range(B):
            masked_positions = torch.where(mask_index[j])[0]
            if masked_positions.numel() == 0:
                continue
            high_conf = (filtered_conf[j] >= _threshold) & mask_index[j]
            if high_conf.any():
                transfer[j] = high_conf
            else:
                # Fallback: single best masked position
                k = min(1, masked_positions.numel())
                _, best = torch.topk(filtered_conf[j], k=k)
                transfer[j, best] = True
        return transfer

    # --- KLASS: confidence threshold + KL-divergence stability ---
    if remasking == "klass":
        _threshold = threshold if threshold is not None else 0.95
        _kl_threshold = kl_threshold if kl_threshold is not None else 0.1
        transfer = torch.zeros(B, L, dtype=torch.bool, device=device)
        for j in range(B):
            masked_positions = torch.where(mask_index[j])[0]
            if masked_positions.numel() == 0:
                continue
            conf_ok = (filtered_conf[j] >= _threshold) & mask_index[j]
            if kl_divergence is not None:
                kl_ok = (kl_divergence[j] <= _kl_threshold) & mask_index[j]
                both_ok = conf_ok & kl_ok
            else:
                # First step: no previous distribution, use confidence only
                both_ok = conf_ok
            if both_ok.any():
                transfer[j] = both_ok
            else:
                # Fallback: single best masked position
                k = min(1, masked_positions.numel())
                _, best = torch.topk(filtered_conf[j], k=k)
                transfer[j, best] = True
        return transfer

    # --- Default (low_confidence, greedy_confidence, probability_margin,
    #     entropy, oracle): top-k highest confidence ---
    transfer = torch.zeros(B, L, dtype=torch.bool, device=device)
    for j in range(B):
        k = int(num_transfer_tokens[j].item())
        if k <= 0:
            continue
        valid_count = (filtered_conf[j] > -float("inf")).sum().item()
        if valid_count == 0:
            continue
        k = min(k, valid_count)
        _, sel = torch.topk(filtered_conf[j], k)
        transfer[j, sel] = True
    return transfer


def compute_kl_divergence(
    prev_probs: torch.Tensor,
    curr_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL(prev || curr) per position.

    Args:
        prev_probs: [B, L, V] probability distribution from previous step.
        curr_probs: [B, L, V] probability distribution from current step.

    Returns:
        kl: [B, L] KL divergence per position.
    """
    eps = 1e-10
    log_ratio = torch.log(prev_probs + eps) - torch.log(curr_probs + eps)
    kl = torch.sum(prev_probs * log_ratio, dim=-1)
    return kl


def oracle_block_enumerate(
    x: torch.Tensor,
    block_start: int,
    block_len: int,
    model: torch.nn.Module,
    temperature: float,
    mask_id: int,
    max_positions: int = 5,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: tuple | None = None,
) -> torch.Tensor:
    """
    Oracle strategy for a single block: enumerate all permutations of masked
    positions, evaluate NLL for each, and return the canvas with the best
    permutation's tokens applied.

    Falls back to greedy confidence when the number of masked positions exceeds
    *max_positions*.

    Args:
        x: [1, T] current canvas (batch size must be 1 for Oracle).
        block_start: start index of the block in the sequence.
        block_len: length of the block.
        model: the language model.
        temperature: gumbel noise temperature.
        mask_id: mask token id.
        max_positions: maximum masked positions to enumerate (default 5).
        attention_mask: optional attention mask for forward pass.
        position_ids: optional position ids for forward pass.
        past_key_values: optional KV cache for prefix.

    Returns:
        x_updated: [1, T] canvas with best-permutation tokens committed.
    """
    B = x.shape[0]
    assert B == 1, "Oracle only supports batch_size=1"

    block_end = block_start + block_len
    block_slice = x[0, block_start:block_end]
    masked_local = torch.where(block_slice == mask_id)[0].tolist()

    if len(masked_local) == 0:
        return x

    # Fallback to greedy if too many positions
    if len(masked_local) > max_positions:
        # Do a single forward pass, greedy unmask all
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
            kwargs["use_cache"] = False
        logits = model(x, **kwargs).logits
        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        mask_index = (x == mask_id)
        x = torch.where(mask_index, x0, x)
        return x

    # Enumerate permutations
    best_nll = float("inf")
    best_tokens = None  # [T] tensor for the whole sequence

    for perm in itertools.permutations(range(len(masked_local))):
        # Work on a copy
        x_trial = x.clone()
        nll = 0.0

        for step_idx in range(len(perm)):
            local_pos = masked_local[perm[step_idx]]
            global_pos = block_start + local_pos

            # Forward pass
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
                kwargs["use_cache"] = False
            logits = model(x_trial, **kwargs).logits

            # Get log probabilities at the target position
            log_p = F.log_softmax(logits[0, global_pos], dim=-1)
            token = torch.argmax(logits[0, global_pos])
            nll -= log_p[token].item()

            # Commit this position
            x_trial[0, global_pos] = token

        if nll < best_nll:
            best_nll = nll
            best_tokens = x_trial[0].clone()

    if best_tokens is not None:
        x = x.clone()
        x[0] = best_tokens

    return x
