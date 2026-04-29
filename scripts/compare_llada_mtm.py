"""Compare original LLaDA sampling against DUEL/MTM variants.

Usage:
    python scripts/compare_llada_mtm.py --config scripts/configs/llada_mtm_compare.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
import yaml

import dllm
from dllm.duel.config import DuelMTMConfig
from dllm.duel.diagnostics import summarise_diagnostics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _normalise_prompts(raw_prompts: list[Any]) -> list[list[dict[str, str]]]:
    prompts = []
    for item in raw_prompts:
        if isinstance(item, str):
            prompts.append([{"role": "user", "content": item}])
        elif isinstance(item, list):
            prompts.append(item)
        else:
            raise TypeError(
                "Each prompt must be either a string or a list of chat messages"
            )
    return prompts


def _make_sampler_config(base_generation: dict[str, Any], run: dict[str, Any]):
    cfg = dict(base_generation)
    cfg.update(run.get("sampler", {}) or {})

    duel_cfg = run.get("duel_mtm")
    if duel_cfg is not None:
        cfg["duel_mtm"] = DuelMTMConfig(**duel_cfg)
    else:
        cfg["duel_mtm"] = None

    return dllm.core.samplers.MDLMSamplerConfig(**cfg)


def _build_inputs(tokenizer, prompts: list[list[dict[str, str]]]):
    return tokenizer.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
    )


def _run_once(
    *,
    sampler,
    tokenizer,
    inputs,
    sampler_config,
    seed: int,
    batch_prompts: list[list[dict[str, str]]],
) -> dict[str, Any]:
    _set_seed(seed)
    _sync_if_cuda()
    start = time.perf_counter()
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
    _sync_if_cuda()
    elapsed = time.perf_counter() - start

    sequences = dllm.utils.sample_trim(
        tokenizer,
        outputs.sequences.tolist(),
        inputs,
    )
    generated_token_counts = [
        max(0, len(seq) - len(inp)) for seq, inp in zip(outputs.sequences.tolist(), inputs)
    ]
    total_generated_tokens = sum(generated_token_counts)

    diagnostics = getattr(outputs, "duel_diagnostics", None)
    diagnostics_summary = summarise_diagnostics(diagnostics) if diagnostics else None

    return {
        "seed": seed,
        "elapsed_sec": elapsed,
        "generated_tokens": generated_token_counts,
        "tokens_per_sec": (
            total_generated_tokens / elapsed if elapsed > 0 and total_generated_tokens else 0.0
        ),
        "outputs": [
            {
                "prompt": prompt,
                "text": text.strip() if text.strip() else "<empty>",
            }
            for prompt, text in zip(batch_prompts, sequences)
        ],
        "duel_diagnostics_summary": diagnostics_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must contain a mapping at the top level")
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    model_cfg = cfg.get("model", {})
    model_name_or_path = model_cfg.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("config.model.model_name_or_path is required")

    print(f"Loading model: {model_name_or_path}")
    model = dllm.utils.get_model(**model_cfg).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_name_or_path)
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

    prompts = _normalise_prompts(cfg.get("prompts", []))
    if not prompts:
        raise ValueError("config.prompts must contain at least one prompt")
    inputs = _build_inputs(tokenizer, prompts)

    base_generation = cfg.get("generation", {})
    runs = cfg.get("runs") or [{"name": "baseline_llada"}]
    output_path = Path(cfg.get("output_path", "outputs/llada_mtm_compare.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": cfg,
        "runs": [],
    }

    for idx, run in enumerate(runs):
        run = dict(run)
        run_name = run.get("name", f"run_{idx}")
        run_seed = int(run.get("seed", seed + idx))
        print(f"\n=== [{idx + 1}/{len(runs)}] {run_name} seed={run_seed} ===")

        sampler_config = _make_sampler_config(base_generation, run)
        result = _run_once(
            sampler=sampler,
            tokenizer=tokenizer,
            inputs=inputs,
            sampler_config=sampler_config,
            seed=run_seed,
            batch_prompts=prompts,
        )
        result["name"] = run_name
        result["sampler_config"] = {
            "generation": base_generation,
            "run": run,
        }
        all_results["runs"].append(result)

        print(
            f"{run_name}: {result['elapsed_sec']:.3f}s, "
            f"{result['tokens_per_sec']:.2f} tok/s"
        )
        for out_idx, item in enumerate(result["outputs"]):
            print(f"\n[{run_name} case {out_idx}]")
            print(item["text"])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
