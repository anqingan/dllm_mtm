"""Compare original LLaDA sampling against DUEL/MTM variants.

Usage:
    python scripts/compare_llada_mtm.py --config scripts/configs/llada_mtm_compare.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
from datasets import load_dataset

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


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _format_template(template: str, row: dict[str, Any]) -> str:
    values = _SafeFormatDict({key: "" if val is None else val for key, val in row.items()})
    return template.format_map(values)


def _load_dataset_items(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = dataset_cfg.get("path")
    if not path:
        raise ValueError("dataset.path is required when dataset.enabled=true")
    kwargs = {}
    for key in ("name", "split", "data_files", "revision"):
        value = dataset_cfg.get(key)
        if value is not None:
            kwargs[key] = value
    if "split" not in kwargs:
        kwargs["split"] = "test"
    dataset = load_dataset(path, **kwargs)
    if dataset_cfg.get("shuffle", False):
        dataset = dataset.shuffle(seed=int(dataset_cfg.get("shuffle_seed", 42)))
    limit = dataset_cfg.get("limit")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    return [dict(row) for row in dataset]


def _dataset_rows_to_prompts_and_targets(
    rows: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
) -> tuple[list[list[dict[str, str]]], list[str | None], list[dict[str, Any]]]:
    prompts: list[list[dict[str, str]]] = []
    targets: list[str | None] = []

    prompt_template = dataset_cfg.get("prompt_template")
    prompt_field = dataset_cfg.get("prompt_field")
    messages_field = dataset_cfg.get("messages_field")
    system_prompt = dataset_cfg.get("system_prompt")
    target_template = dataset_cfg.get("target_template")
    target_field = dataset_cfg.get("target_field")

    for row in rows:
        if messages_field:
            messages = row.get(messages_field)
            if isinstance(messages, str):
                messages = json.loads(messages)
            if not isinstance(messages, list):
                raise ValueError(f"messages_field={messages_field!r} must contain a list")
            prompt_messages = messages
        else:
            if prompt_template:
                user_content = _format_template(prompt_template, row)
            elif prompt_field:
                user_content = str(row.get(prompt_field, ""))
            else:
                raise ValueError(
                    "dataset.prompt_template, dataset.prompt_field, or "
                    "dataset.messages_field is required"
                )
            prompt_messages = []
            if system_prompt:
                prompt_messages.append({"role": "system", "content": str(system_prompt)})
            prompt_messages.append({"role": "user", "content": user_content})
        prompts.append(prompt_messages)

        if target_template:
            targets.append(_format_template(target_template, row))
        elif target_field:
            value = row.get(target_field)
            targets.append(None if value is None else str(value))
        else:
            targets.append(None)

    return prompts, targets, rows


def _normalise_for_score(text: str, *, case_sensitive: bool, strip: bool) -> str:
    if strip:
        text = text.strip()
    return text if case_sensitive else text.lower()


def _extract_last_number(text: str) -> str | None:
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _score_prediction(prediction: str, target: str | None, scoring_cfg: dict[str, Any]) -> dict[str, Any]:
    metric = scoring_cfg.get("metric", "none")
    if metric == "none" or target is None:
        return {"score": None, "correct": None}

    case_sensitive = bool(scoring_cfg.get("case_sensitive", False))
    strip = bool(scoring_cfg.get("strip", True))
    pred = _normalise_for_score(prediction, case_sensitive=case_sensitive, strip=strip)
    gold = _normalise_for_score(target, case_sensitive=case_sensitive, strip=strip)

    if metric == "exact_match":
        correct = pred == gold
    elif metric == "contains":
        correct = gold in pred
    elif metric == "regex":
        pattern = scoring_cfg.get("regex")
        if not pattern:
            raise ValueError("scoring.regex is required when metric='regex'")
        flags = 0 if case_sensitive else re.IGNORECASE
        match = re.search(pattern, prediction, flags=flags)
        if scoring_cfg.get("regex_compare_target", False):
            correct = bool(match and match.group(1).strip() == target.strip())
        else:
            correct = bool(match)
    elif metric == "gsm8k_number":
        correct = _extract_last_number(prediction) == _extract_last_number(target)
    else:
        raise ValueError(f"Unknown scoring.metric: {metric!r}")

    return {"score": 1.0 if correct else 0.0, "correct": bool(correct)}


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


def _iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _run_once(
    *,
    sampler,
    tokenizer,
    sampler_config,
    seed: int,
    batch_prompts: list[list[dict[str, str]]],
    targets: list[str | None] | None,
    rows: list[dict[str, Any]] | None,
    scoring_cfg: dict[str, Any],
    eval_batch_size: int,
) -> dict[str, Any]:
    _set_seed(seed)
    all_sequences: list[str] = []
    all_generated_token_counts: list[int] = []
    all_diagnostics = []
    elapsed = 0.0

    for start_idx, prompt_batch in _iter_batches(batch_prompts, eval_batch_size):
        input_batch = _build_inputs(tokenizer, prompt_batch)
        _sync_if_cuda()
        start = time.perf_counter()
        outputs = sampler.sample(input_batch, sampler_config, return_dict=True)
        _sync_if_cuda()
        elapsed += time.perf_counter() - start

        batch_sequences = dllm.utils.sample_trim(
            tokenizer,
            outputs.sequences.tolist(),
            input_batch,
        )
        all_sequences.extend(batch_sequences)
        all_generated_token_counts.extend(
            [
                max(0, len(seq) - len(inp))
                for seq, inp in zip(outputs.sequences.tolist(), input_batch)
            ]
        )

        diagnostics = getattr(outputs, "duel_diagnostics", None)
        if diagnostics:
            all_diagnostics.extend(diagnostics)

    total_generated_tokens = sum(all_generated_token_counts)
    diagnostics_summary = summarise_diagnostics(all_diagnostics) if all_diagnostics else None

    scored_outputs = []
    scores = []
    targets = targets or [None] * len(batch_prompts)
    rows = rows or [{} for _ in batch_prompts]
    for idx, (prompt, text, target, row) in enumerate(
        zip(batch_prompts, all_sequences, targets, rows)
    ):
        text = text.strip() if text.strip() else "<empty>"
        score = _score_prediction(text, target, scoring_cfg)
        if score["score"] is not None:
            scores.append(float(score["score"]))
        scored_outputs.append(
            {
                "index": idx,
                "prompt": prompt,
                "target": target,
                "prediction": text,
                "score": score,
                "row": row if scoring_cfg.get("include_rows", False) else None,
            }
        )

    aggregate_score = sum(scores) / len(scores) if scores else None

    return {
        "seed": seed,
        "elapsed_sec": elapsed,
        "generated_tokens": all_generated_token_counts,
        "tokens_per_sec": (
            total_generated_tokens / elapsed if elapsed > 0 and total_generated_tokens else 0.0
        ),
        "metric": scoring_cfg.get("metric", "none"),
        "score": aggregate_score,
        "num_scored": len(scores),
        "outputs": scored_outputs,
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

    dataset_cfg = cfg.get("dataset", {}) or {}
    if dataset_cfg.get("enabled", False):
        rows = _load_dataset_items(dataset_cfg)
        prompts, targets, rows = _dataset_rows_to_prompts_and_targets(rows, dataset_cfg)
    else:
        prompts = _normalise_prompts(cfg.get("prompts", []))
        targets = [None] * len(prompts)
        rows = [{} for _ in prompts]
    if not prompts:
        raise ValueError("No prompts found. Provide config.prompts or enable config.dataset")

    base_generation = cfg.get("generation", {})
    scoring_cfg = cfg.get("scoring", {}) or {}
    eval_batch_size = int(cfg.get("eval_batch_size", 1))
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
            sampler_config=sampler_config,
            seed=run_seed,
            batch_prompts=prompts,
            targets=targets,
            rows=rows,
            scoring_cfg=scoring_cfg,
            eval_batch_size=eval_batch_size,
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
        if result["score"] is not None:
            print(f"{run_name}: {result['metric']}={result['score']:.4f} ({result['num_scored']} examples)")
        for out_idx, item in enumerate(result["outputs"]):
            print(f"\n[{run_name} case {out_idx}]")
            if item["target"] is not None:
                print(f"Target: {item['target']}")
            print(item["prediction"])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
