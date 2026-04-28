"""
DUEL-guided Intra-Block MTM demo script.

Demonstrates baseline, DUEL-Rerank, and DUEL-Intra-MTM generation
using a toy fake model (no real checkpoint needed).

Usage:
    cd /path/to/dllm && python scripts/run_duel_intra_mtm_demo.py
"""

import importlib
import importlib.util
import os
import sys
import types

# Set up minimal mock for dllm so we can import dllm.duel without
# triggering the full pipeline import cascade.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_dllm_path = os.path.join(_project_root, "dllm")

# Create a minimal dllm package that doesn't auto-import pipelines
dllm_pkg = types.ModuleType("dllm")
dllm_pkg.__path__ = [_dllm_path]
dllm_pkg.__package__ = "dllm"
sys.modules["dllm"] = dllm_pkg

duel_pkg = types.ModuleType("dllm.duel")
duel_pkg.__path__ = [os.path.join(_dllm_path, "duel")]
duel_pkg.__package__ = "dllm.duel"
sys.modules["dllm.duel"] = duel_pkg


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_dllm_path, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("dllm.duel.unmasking", "duel/unmasking.py")
_load("dllm.duel.config", "duel/config.py")
_load("dllm.duel.types", "duel/types.py")
_load("dllm.duel.duel_scorer", "duel/duel_scorer.py")
_load("dllm.duel.duel_sampler", "duel/duel_sampler.py")
_load("dllm.duel.block_utils", "duel/block_utils.py")
_load("dllm.duel.diagnostics", "duel/diagnostics.py")
_load("dllm.duel.intra_block_mtm", "duel/intra_block_mtm.py")

import torch
import torch.nn as nn

# ── Fake model for demo ──────────────────────────────────────────────────


class FakeDLLM(nn.Module):
    """A minimal masked-LM that returns random-ish logits."""

    def __init__(self, vocab_size: int = 50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 16)
        self.head = nn.Linear(16, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        logits = self.head(h)
        return type("Out", (), {"logits": logits})()


# ── Main demo ────────────────────────────────────────────────────────────


def main():
    from dllm.duel.block_utils import make_blocks
    from dllm.duel.config import DuelMTMConfig
    from dllm.duel.diagnostics import summarise_diagnostics
    from dllm.duel.duel_sampler import duel_generate_region
    from dllm.duel.intra_block_mtm import IntraBlockDuelMTM

    torch.manual_seed(0)
    MASK_ID = 0
    VOCAB = 50
    T = 16
    BLOCK_SIZE = 8

    model = FakeDLLM(vocab_size=VOCAB).eval()

    # Fake tokenizer attributes (demo only)
    class FakeTok:
        mask_token_id = MASK_ID
        eos_token_id = 2
        pad_token_id = 1
        bos_token_id = 3

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    tokenizer = FakeTok()

    # Prompt: [3, 4, 5], generate positions 3..15
    prompt = torch.tensor([[3, 4, 5] + [MASK_ID] * (T - 3)])
    attention_mask = torch.ones(1, T, dtype=torch.long)

    print("=" * 70)
    print("DUEL-guided Intra-Block MTM Demo")
    print("=" * 70)

    # ── 1. Baseline generation ──────────────────────────────────────────
    print("\n[1] Baseline DUEL generation (no MTM)")
    gen_mask = (prompt == MASK_ID)
    gen = torch.Generator().manual_seed(42)

    baseline_state, trace = duel_generate_region(
        model=model,
        initial_state=prompt,
        generation_mask=gen_mask,
        unmask_rule="probability_margin",
        positions_per_step=1,
        mask_token_id=MASK_ID,
        temperature=1.0,
        attention_mask=attention_mask,
        generator=gen,
        return_trace=True,
    )
    print(f"Baseline output: {baseline_state[0].tolist()}")
    print(f"Generation trace steps: {len(trace['steps']) if trace else 'N/A'}")

    # ── 2. DUEL-Rerank ─────────────────────────────────────────────────
    print("\n[2] DUEL-Rerank")
    cfg_rerank = DuelMTMConfig(
        enabled=True,
        method="duel_rerank",
        K=4,
        beta=4.0,
        block_size=BLOCK_SIZE,
        rollback_ratio=0.5,
        unmask_rule="probability_margin",
        positions_per_step=1,
        temperature=1.0,
    )

    state = baseline_state.clone()
    mtm_rerank = IntraBlockDuelMTM(model, cfg_rerank, MASK_ID, tokenizer)
    blocks = make_blocks(T, BLOCK_SIZE)
    all_diag_rerank = []

    for bi, (bs, be) in enumerate(blocks):
        gen = torch.Generator().manual_seed(42 + bi)
        state, diag = mtm_rerank.step(
            state, block_start=bs, block_end=be, block_index=bi,
            attention_mask=attention_mask, generator=gen,
        )
        all_diag_rerank.append(diag)

    print(f"Rerank output:  {state[0].tolist()}")
    summary = summarise_diagnostics(all_diag_rerank)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ── 3. DUEL-Intra-MTM ──────────────────────────────────────────────
    print("\n[3] DUEL-Intra-MTM")
    cfg_mtm = DuelMTMConfig(
        enabled=True,
        method="duel_mtm",
        K=4,
        beta=4.0,
        block_size=BLOCK_SIZE,
        rollback_ratio=0.5,
        unmask_rule="probability_margin",
        positions_per_step=1,
        temperature=1.0,
    )

    state = baseline_state.clone()
    mtm = IntraBlockDuelMTM(model, cfg_mtm, MASK_ID, tokenizer)
    all_diag_mtm = []

    for bi, (bs, be) in enumerate(blocks):
        for _step in range(cfg_mtm.num_mtm_steps_per_block):
            gen = torch.Generator().manual_seed(42 + bi)
            state, diag = mtm.step(
                state, block_start=bs, block_end=be, block_index=bi,
                attention_mask=attention_mask, generator=gen,
            )
            all_diag_mtm.append(diag)

    print(f"MTM output:     {state[0].tolist()}")
    summary = summarise_diagnostics(all_diag_mtm)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("Demo complete. No inter-block MTM was used.")
    print("=" * 70)


if __name__ == "__main__":
    main()
