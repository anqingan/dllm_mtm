"""Tests for DUEL block and rollback utilities."""

import torch

from dllm.duel.block_utils import make_block_mask, make_blocks, make_rollback_state

MASK_ID = 0


def test_make_blocks_covers_sequence():
    assert make_blocks(7, 3) == [(0, 3), (3, 6), (6, 7)]


def test_rollback_only_masks_current_block():
    state = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    block_mask = make_block_mask(8, 2, 6)

    rollback_state, rollback_mask, preserved_mask = make_rollback_state(
        current_state=state,
        block_mask=block_mask,
        policy="fixed_ratio",
        rollback_ratio=0.5,
        mask_token_id=MASK_ID,
    )

    assert torch.equal(rollback_state[0, :2], state[0, :2])
    assert torch.equal(rollback_state[0, 6:], state[0, 6:])
    assert rollback_mask.tolist() == [[False, False, False, False, True, True, False, False]]
    assert preserved_mask.tolist() == [[False, False, True, True, False, False, False, False]]
    assert rollback_state[0, 4].item() == MASK_ID
    assert rollback_state[0, 5].item() == MASK_ID
