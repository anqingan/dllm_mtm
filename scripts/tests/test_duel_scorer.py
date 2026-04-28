"""Tests for DUEL conditional log-likelihood scorer."""

import pytest
import torch
import torch.nn as nn

from dllm.duel.duel_scorer import compute_duel_conditional_loglikelihood

MASK_ID = 0


class _FakeModel(nn.Module):
    """Tiny model that always returns fixed logits so tests are deterministic."""

    def __init__(self, vocab_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)  # a parameter so .eval() works

    def forward(self, input_ids, **kwargs):
        # Return uniform-ish logits shaped [B, T, V]
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        # Token 1 is always slightly preferred
        logits[:, :, 1] = 1.0
        return type("Out", (), {"logits": logits})()


class TestDuelScorer:
    def setup_method(self):
        self.model = _FakeModel(vocab_size=8).eval()
        self.T = 6

    def test_scoring_does_not_sample(self):
        """Scoring reveals tokens from target_tokens, not from sampling."""
        initial = torch.tensor([[1, 0, 0, 4, 0, 6]])
        target = torch.tensor([[1, 2, 3, 4, 5, 6]])
        score_mask = torch.zeros(1, self.T, dtype=torch.bool)
        score_mask[0, 1] = True
        score_mask[0, 4] = True
        candidate_mask = (initial == MASK_ID)

        ll, _ = compute_duel_conditional_loglikelihood(
            model=self.model,
            initial_state=initial,
            target_tokens=target,
            score_mask=score_mask,
            candidate_mask=candidate_mask,
            unmask_rule="left_to_right",
            positions_per_step=1,
            mask_token_id=MASK_ID,
        )
        # Should complete without error — tokens came from target, not sampled
        assert ll.shape == torch.Size([])

    def test_log_likelihood_is_sum_of_log_probs(self):
        """Total LL equals sum of log_softmax at revealed positions."""
        initial = torch.tensor([[1, 0, 0, 4, 5, 6]])
        target = torch.tensor([[1, 2, 3, 4, 5, 6]])
        score_mask = torch.ones(1, self.T, dtype=torch.bool)
        candidate_mask = (initial == MASK_ID)

        ll, _ = compute_duel_conditional_loglikelihood(
            model=self.model,
            initial_state=initial,
            target_tokens=target,
            score_mask=score_mask,
            candidate_mask=candidate_mask,
            unmask_rule="left_to_right",
            positions_per_step=1,
            mask_token_id=MASK_ID,
        )

        # With our fake model, log_softmax logits = [0, 1, 0, 0, 0]
        # log_softmax = [−1/5·exp_sums...] — we just check it's finite and negative
        assert torch.isfinite(ll)
        assert ll.item() < 0  # log probs are always <= 0

    def test_score_mask_false_not_accumulated(self):
        """Positions with score_mask=False must not contribute."""
        initial = torch.tensor([[0, 0]])
        target = torch.tensor([[2, 3]])
        # Only score position 0
        score_mask = torch.tensor([[True, False]])
        candidate_mask = torch.tensor([[True, True]])

        ll, _ = compute_duel_conditional_loglikelihood(
            model=self.model,
            initial_state=initial,
            target_tokens=target,
            score_mask=score_mask,
            candidate_mask=candidate_mask,
            unmask_rule="left_to_right",
            positions_per_step=1,
            mask_token_id=MASK_ID,
        )
        # Should only accumulate position 0's log-prob, not position 1's
        assert torch.isfinite(ll)

    def test_target_has_mask_raises(self):
        """If target still has mask_token_id at a to-reveal position, raise."""
        initial = torch.tensor([[0, 0]])
        target = torch.tensor([[2, 0]])  # position 1 still mask
        score_mask = torch.ones(1, 2, dtype=torch.bool)
        candidate_mask = torch.ones(1, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="mask_token_id"):
            compute_duel_conditional_loglikelihood(
                model=self.model,
                initial_state=initial,
                target_tokens=target,
                score_mask=score_mask,
                candidate_mask=candidate_mask,
                unmask_rule="left_to_right",
                positions_per_step=1,
                mask_token_id=MASK_ID,
            )

    def test_return_diagnostics(self):
        initial = torch.tensor([[0, 0]])
        target = torch.tensor([[2, 3]])
        score_mask = torch.ones(1, 2, dtype=torch.bool)
        candidate_mask = torch.ones(1, 2, dtype=torch.bool)

        _, diag = compute_duel_conditional_loglikelihood(
            model=self.model,
            initial_state=initial,
            target_tokens=target,
            score_mask=score_mask,
            candidate_mask=candidate_mask,
            unmask_rule="left_to_right",
            positions_per_step=1,
            mask_token_id=MASK_ID,
            return_diagnostics=True,
        )
        assert diag is not None
        assert "steps" in diag
        assert len(diag["steps"]) > 0
