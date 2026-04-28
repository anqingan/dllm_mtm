"""Tests for DUEL region generation."""

import torch
import torch.nn as nn

from dllm.duel.duel_sampler import duel_generate_region

MASK_ID = 0


class _FakeModel(nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 1] = 4.0
        logits[:, :, 2] = 2.0
        return type("Out", (), {"logits": logits})()


def test_duel_sampler_only_generates_region():
    model = _FakeModel().eval()
    initial = torch.tensor([[9, 0, 0, 8, 0]])
    generation_mask = torch.tensor([[False, True, False, False, True]])
    gen = torch.Generator().manual_seed(0)

    result, _ = duel_generate_region(
        model=model,
        initial_state=initial,
        generation_mask=generation_mask,
        unmask_rule="left_to_right",
        positions_per_step=1,
        mask_token_id=MASK_ID,
        generator=gen,
    )

    assert result[0, 0].item() == 9
    assert result[0, 2].item() == MASK_ID
    assert result[0, 3].item() == 8
    assert result[0, 1].item() != MASK_ID
    assert result[0, 4].item() != MASK_ID


def test_duel_sampler_reproducible_with_seed():
    model = _FakeModel().eval()
    initial = torch.tensor([[0, 0, 0]])
    generation_mask = initial == MASK_ID

    out = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(123)
        result, _ = duel_generate_region(
            model=model,
            initial_state=initial,
            generation_mask=generation_mask,
            unmask_rule="probability_margin",
            positions_per_step=1,
            mask_token_id=MASK_ID,
            generator=gen,
        )
        out.append(result)

    assert torch.equal(out[0], out[1])
