"""Block utilities for DUEL-guided Intra-Block MTM.

Provides helpers for splitting sequences into blocks, constructing block
masks, and creating rollback states.
"""

from __future__ import annotations

import math

import torch


def make_blocks(seq_len: int, block_size: int) -> list[tuple[int, int]]:
    """Return a list of (start, end) tuples covering [0, seq_len) in blocks."""
    blocks = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        blocks.append((start, end))
    return blocks


def make_block_mask(
    seq_len: int,
    block_start: int,
    block_end: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a [1, seq_len] bool tensor that is True inside [block_start, block_end)."""
    mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    mask[0, block_start:block_end] = True
    return mask


def make_rollback_state(
    current_state: torch.Tensor,  # [1, T]
    block_mask: torch.Tensor,  # [1, T] bool – current block positions
    policy: str,
    rollback_ratio: float,
    generation_trace: dict | None = None,
    mask_token_id: int = -1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a rollback state by re-masking some tokens in the current block.

    Returns (rollback_state, rollback_mask, preserved_mask):
      - rollback_state: [1, T] copy of current_state with selected tokens re-masked
      - rollback_mask:  [1, T] bool – positions that were re-masked
      - preserved_mask: [1, T] bool – positions in the block that were kept

    Only tokens inside block_mask are ever re-masked; tokens outside the block
    are never modified.
    """
    z = current_state.clone()
    B, T = z.shape

    block_positions = torch.where(block_mask[0])[0]  # indices inside block
    if block_positions.numel() == 0:
        # Empty block — nothing to roll back
        return z, torch.zeros_like(block_mask), torch.zeros_like(block_mask)

    # Identify already-generated (non-mask) tokens in the block
    generated_mask = block_mask[0] & (current_state[0] != mask_token_id)
    generated_positions = torch.where(generated_mask)[0]

    if generated_positions.numel() == 0:
        # Nothing generated yet — nothing to roll back
        return z, torch.zeros_like(block_mask), block_mask & generated_mask.logical_not()

    if policy == "fixed_ratio":
        num_to_mask = max(1, math.ceil(rollback_ratio * block_positions.numel()))
        # Re-mask the *last* num_to_mask positions in the block (rightmost)
        to_mask_positions = generated_positions[-num_to_mask:]
    elif policy == "uniform_step":
        # If generation_trace is available, pick a random step and re-mask
        # everything revealed after that step.  For now, fall back to fixed_ratio.
        if generation_trace is not None and "steps" in generation_trace:
            steps = generation_trace["steps"]
            if len(steps) > 1:
                step_idx = torch.randint(
                    1, len(steps), (1,), generator=generator
                ).item()
                # Collect all positions revealed from step_idx onward
                to_mask_positions_list = []
                for s in steps[step_idx:]:
                    to_mask_positions_list.extend(s["positions"])
                to_mask_positions = torch.tensor(
                    to_mask_positions_list, dtype=torch.long, device=z.device
                )
                # Intersect with generated positions
                gen_set = set(generated_positions.tolist())
                to_mask_positions = torch.tensor(
                    [p for p in to_mask_positions.tolist() if p in gen_set],
                    dtype=torch.long,
                    device=z.device,
                )
            else:
                num_to_mask = max(
                    1, math.ceil(rollback_ratio * block_positions.numel())
                )
                to_mask_positions = generated_positions[-num_to_mask:]
        else:
            num_to_mask = max(
                1, math.ceil(rollback_ratio * block_positions.numel())
            )
            to_mask_positions = generated_positions[-num_to_mask:]
    else:
        raise ValueError(f"Unknown rollback_policy: {policy!r}")

    # Build masks
    rollback_mask = torch.zeros(1, T, dtype=torch.bool, device=z.device)
    if to_mask_positions.numel() > 0:
        rollback_mask[0, to_mask_positions] = True

    preserved_mask = block_mask & rollback_mask.logical_not() & block_mask

    # Apply re-masking
    z[rollback_mask] = mask_token_id

    return z, rollback_mask, preserved_mask
