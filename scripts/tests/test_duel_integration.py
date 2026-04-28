"""Integration tests for optional DUEL-MTM in the MDLM sampler."""

import torch
import torch.nn as nn

from dllm.core.samplers import DuelMTMConfig, MDLMSampler, MDLMSamplerConfig


class _FakeTokenizer:
    mask_token_id = 0
    eos_token_id = 2
    bos_token_id = 3
    pad_token_id = 2


class _FakeModel(nn.Module):
    device = torch.device("cpu")

    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 1] = 5.0
        logits[:, :, 4] = 3.0
        return type("Out", (), {"logits": logits})()


def _sample_with(duel_mtm):
    sampler = MDLMSampler(model=_FakeModel().eval(), tokenizer=_FakeTokenizer())
    cfg = MDLMSamplerConfig(
        max_new_tokens=4,
        block_size=2,
        steps=4,
        temperature=0.0,
        remasking="left_to_right",
        return_dict=True,
        duel_mtm=duel_mtm,
    )
    return sampler.sample([[3, 4]], cfg)


def test_duel_disabled_matches_baseline_path():
    baseline = _sample_with(None)
    disabled = _sample_with(DuelMTMConfig(enabled=False))

    assert torch.equal(baseline.sequences, disabled.sequences)
    assert not hasattr(disabled, "duel_diagnostics")


def test_duel_enabled_records_diagnostics():
    enabled = _sample_with(
            DuelMTMConfig(
                enabled=True,
                method="duel_rerank",
                exact_mtm=False,
                K=2,
            block_size=2,
            rollback_ratio=0.5,
            unmask_rule="left_to_right",
            positions_per_step=1,
            seed=123,
        )
    )

    assert hasattr(enabled, "duel_diagnostics")
    assert len(enabled.duel_diagnostics) == 2
    assert all(d.method == "duel_rerank" for d in enabled.duel_diagnostics)
