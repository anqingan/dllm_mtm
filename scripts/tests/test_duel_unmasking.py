"""Tests for DUEL deterministic unmasking rules."""

import pytest
import torch
import torch.nn.functional as F

from dllm.duel.unmasking import select_positions

MASK_ID = 0


class TestSelectPositionsBasic:
    """Core correctness: deterministic, no non-mask selection, tie-breaking."""

    def _make_inputs(self, T=10):
        # Positions 2, 5, 8 are masked (token 0); others are filled (1..V)
        input_ids = torch.tensor([[1, 3, 0, 4, 5, 0, 6, 7, 0, 9]])
        candidate_mask = (input_ids == MASK_ID)
        logits = torch.randn(1, T, 100)
        return input_ids, candidate_mask, logits

    @pytest.mark.parametrize("rule", [
        "left_to_right",
        "greedy_confidence",
        "probability_margin",
    ])
    def test_deterministic(self, rule):
        ids, cmask, logits = self._make_inputs()
        r1 = select_positions(ids, logits, cmask, rule, 2, MASK_ID)
        r2 = select_positions(ids, logits, cmask, rule, 2, MASK_ID)
        assert r1 == r2

    @pytest.mark.parametrize("rule", [
        "left_to_right",
        "greedy_confidence",
        "probability_margin",
    ])
    def test_never_selects_non_mask(self, rule):
        ids, cmask, logits = self._make_inputs()
        result = select_positions(ids, logits, cmask, rule, 3, MASK_ID)
        for pos in result[0]:
            assert cmask[0, pos].item() is True

    @pytest.mark.parametrize("rule", [
        "left_to_right",
        "greedy_confidence",
        "probability_margin",
    ])
    def test_never_selects_candidate_false(self, rule):
        ids, _, logits = self._make_inputs()
        # Only position 5 is candidate
        restricted = torch.zeros_like(ids, dtype=torch.bool)
        restricted[0, 5] = True
        result = select_positions(ids, logits, restricted, rule, 5, MASK_ID)
        for pos in result[0]:
            assert pos == 5

    def test_left_to_right_order(self):
        ids, cmask, logits = self._make_inputs()
        result = select_positions(ids, logits, cmask, "left_to_right", 2, MASK_ID)
        # Masked positions are 2, 5, 8 → leftmost two are 2, 5
        assert result[0] == [2, 5]

    def test_left_to_right_k1(self):
        ids, cmask, logits = self._make_inputs()
        result = select_positions(ids, logits, cmask, "left_to_right", 1, MASK_ID)
        assert result[0] == [2]

    def test_fewer_positions_than_k(self):
        ids, _, logits = self._make_inputs()
        cmask = torch.zeros(1, 10, dtype=torch.bool)
        cmask[0, 5] = True  # only one masked candidate
        result = select_positions(ids, logits, cmask, "left_to_right", 5, MASK_ID)
        assert result[0] == [5]

    def test_no_candidates(self):
        ids = torch.tensor([[1, 2, 3]])
        cmask = torch.zeros(1, 3, dtype=torch.bool)
        logits = torch.randn(1, 3, 10)
        result = select_positions(ids, logits, cmask, "left_to_right", 2, MASK_ID)
        assert result[0] == []

    def test_greedy_confidence_selects_highest_prob(self):
        ids = torch.tensor([[0, 0]])
        cmask = torch.ones(1, 2, dtype=torch.bool)
        logits = torch.zeros(1, 2, 5)
        # Position 0: uniform → top1 prob = 0.2
        # Position 1: peaked at token 0 → top1 prob = 0.8
        logits[0, 1, 0] = 10.0
        result = select_positions(ids, logits, cmask, "greedy_confidence", 1, MASK_ID)
        assert result[0] == [1]

    def test_probability_margin_selects_largest_gap(self):
        ids = torch.tensor([[0, 0]])
        cmask = torch.ones(1, 2, dtype=torch.bool)
        logits = torch.zeros(1, 2, 5)
        # Position 0: top1=0.4, top2=0.3 → margin=0.1
        logits[0, 0, 0] = 1.0
        logits[0, 0, 1] = 0.7
        # Position 1: top1=0.5, top2=0.1 → margin=0.4
        logits[0, 1, 0] = 2.0
        logits[0, 1, 1] = 0.0
        result = select_positions(ids, logits, cmask, "probability_margin", 1, MASK_ID)
        assert result[0] == [1]

    def test_unknown_rule_raises(self):
        ids = torch.tensor([[0]])
        cmask = torch.ones(1, 1, dtype=torch.bool)
        logits = torch.randn(1, 1, 10)
        with pytest.raises(ValueError, match="Unknown unmask_rule"):
            select_positions(ids, logits, cmask, "bogus", 1, MASK_ID)
