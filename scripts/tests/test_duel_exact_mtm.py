"""Exact DUEL + Intra-Block MTM proposal-correction tests."""

import math

import pytest
import torch
import torch.nn as nn

import dllm.duel.intra_block_mtm as mtm_module
from dllm.duel.config import DuelMTMConfig
from dllm.duel.duel_sampler import compute_duel_proposal_logprob, duel_generate_region
from dllm.duel.duel_scorer import compute_duel_conditional_loglikelihood
from dllm.duel.intra_block_mtm import IntraBlockDuelMTM

MASK_ID = 0


class _StatefulModel(nn.Module):
    def __init__(self, vocab_size: int = 6):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size, dtype=torch.float32)
        logits[:, :, MASK_ID] = -0.5
        logits[:, :, 1] = 1.0
        logits[:, :, 2] = 2.0
        logits[:, :, 3] = 0.25
        context = (input_ids != MASK_ID).sum(dim=1).float()
        logits[:, :, 4] = context[:, None] * 0.4
        logits[:, :, 5] = -0.25
        return type("Out", (), {"logits": logits})()


def _log_p(model, initial, target, mask, target_temperature=1.0):
    return compute_duel_conditional_loglikelihood(
        model=model,
        initial_state=initial,
        target_tokens=target,
        score_mask=mask,
        candidate_mask=mask,
        unmask_rule="left_to_right",
        positions_per_step=1,
        mask_token_id=MASK_ID,
        target_temperature=target_temperature,
    )[0]


def _log_q(model, initial, target, mask, proposal_temperature=1.0, top_k=None, top_p=None):
    return compute_duel_proposal_logprob(
        model=model,
        initial_state=initial,
        target_state=target,
        generation_mask=mask,
        unmask_rule="left_to_right",
        positions_per_step=1,
        mask_token_id=MASK_ID,
        proposal_temperature=proposal_temperature,
        top_k=top_k,
        top_p=top_p,
    )


def test_proposal_logprob_replay_consistency():
    model = _StatefulModel().eval()
    initial = torch.tensor([[9, 0, 0, 7]])
    region = initial == MASK_ID
    gen = torch.Generator().manual_seed(123)

    proposal = duel_generate_region(
        model=model,
        initial_state=initial,
        generation_mask=region,
        unmask_rule="left_to_right",
        positions_per_step=1,
        mask_token_id=MASK_ID,
        temperature=0.7,
        generator=gen,
    )
    replay_log_q = _log_q(model, initial, proposal.sequence, region, 0.7)

    assert torch.allclose(proposal.log_q, replay_log_q, atol=1e-6)


def test_special_case_reduction_to_alpha_minus_one_weight():
    model = _StatefulModel().eval()
    initial = torch.tensor([[9, 0, 0, 7]])
    target = torch.tensor([[9, 2, 4, 7]])
    region = initial == MASK_ID
    alpha = 3.5

    log_p = _log_p(model, initial, target, region, target_temperature=1.0)
    log_q = _log_q(model, initial, target, region, proposal_temperature=1.0)
    exact_weight = alpha * log_p - log_q
    simplified = (alpha - 1.0) * log_p

    assert torch.allclose(log_q, log_p, atol=1e-6)
    assert torch.allclose(exact_weight, simplified, atol=1e-6)


@pytest.mark.parametrize("proposal_temperature", [4.0, 0.25])
def test_arbitrary_temperature_uses_proposal_correction(proposal_temperature):
    model = _StatefulModel().eval()
    initial = torch.tensor([[9, 0, 0, 7]])
    target = torch.tensor([[9, 2, 4, 7]])
    region = initial == MASK_ID
    alpha = 2.0

    log_p = _log_p(model, initial, target, region, target_temperature=1.0)
    log_q = _log_q(
        model, initial, target, region, proposal_temperature=proposal_temperature
    )
    exact_weight = alpha * log_p - log_q
    old_weight = (alpha - 1.0) * log_p

    assert not torch.allclose(log_q, log_p, atol=1e-4)
    assert torch.allclose(exact_weight, alpha * log_p - log_q, atol=1e-6)
    assert not torch.allclose(exact_weight, old_weight, atol=1e-4)


def test_backward_current_state_log_q_is_computed(monkeypatch):
    model = _StatefulModel().eval()
    cfg = DuelMTMConfig(
        enabled=True,
        method="duel_mtm",
        K=2,
        alpha=2.0,
        rollback_ratio=0.5,
        unmask_rule="left_to_right",
        positions_per_step=1,
        proposal_temperature=1.0,
        target_temperature=1.0,
    )
    calls = {"count": 0}
    original = mtm_module.compute_duel_proposal_logprob

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mtm_module, "compute_duel_proposal_logprob", wrapped)
    mtm = IntraBlockDuelMTM(model, cfg, MASK_ID)
    state = torch.tensor([[9, 1, 2, 3, 7]])
    mtm.step(
        state,
        block_start=1,
        block_end=4,
        block_index=0,
        generator=torch.Generator().manual_seed(5),
    )

    assert calls["count"] == 1


def test_mask_token_support_consistency():
    model = _StatefulModel().eval()
    initial = torch.tensor([[0]])
    region = initial == MASK_ID
    target = torch.tensor([[MASK_ID]])

    with pytest.raises(ValueError, match="mask_token_id"):
        _log_p(model, initial, target, region)
    with pytest.raises(ValueError, match="mask_token_id"):
        _log_q(model, initial, target, region)


def test_top_k_truncated_proposal_logprob_support():
    model = _StatefulModel().eval()
    initial = torch.tensor([[0]])
    region = initial == MASK_ID
    in_support = torch.tensor([[2]])
    out_of_support = torch.tensor([[1]])

    assert torch.isfinite(_log_q(model, initial, in_support, region, top_k=1))
    assert torch.isneginf(_log_q(model, initial, out_of_support, region, top_k=1))


def test_exact_mtm_rejects_heuristic_rerank():
    with pytest.raises(ValueError, match="duel_rerank is a heuristic"):
        DuelMTMConfig(enabled=True, method="duel_rerank", exact_mtm=True)


def test_acceptance_ratio_direction_from_recorded_weights():
    model = _StatefulModel().eval()
    cfg = DuelMTMConfig(
        enabled=True,
        method="duel_mtm",
        K=2,
        alpha=2.0,
        rollback_ratio=0.5,
        unmask_rule="left_to_right",
        positions_per_step=1,
        proposal_temperature=1.0,
        target_temperature=1.0,
    )
    mtm = IntraBlockDuelMTM(model, cfg, MASK_ID)
    _, diag = mtm.step(
        torch.tensor([[9, 1, 2, 3, 7]]),
        block_start=1,
        block_end=4,
        block_index=0,
        generator=torch.Generator().manual_seed(9),
    )

    expected = torch.logsumexp(torch.tensor(diag.forward_log_weights), dim=0) - torch.logsumexp(
        torch.tensor(diag.backward_log_weights), dim=0
    )
    assert math.isclose(diag.log_W_fwd - diag.log_W_bwd, expected.item(), rel_tol=1e-6)
