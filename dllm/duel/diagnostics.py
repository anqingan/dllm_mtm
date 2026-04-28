"""Diagnostics utilities for DUEL-guided Intra-Block MTM.

Provides helpers for collecting, saving, and summarising diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

from dllm.duel.types import DuelDiagnostics


def save_diagnostics(
    diagnostics: list[DuelDiagnostics],
    path: str | Path,
) -> None:
    """Append diagnostics records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for d in diagnostics:
            f.write(json.dumps(d.to_dict()) + "\n")


def summarise_diagnostics(diagnostics: list[DuelDiagnostics]) -> dict:
    """Return a summary dict over a list of diagnostics."""
    if not diagnostics:
        return {}

    n = len(diagnostics)
    accept_probs = [d.accept_prob for d in diagnostics]
    accepted = [d.accepted for d in diagnostics]
    fwd_lls = [d.forward_loglikelihoods for d in diagnostics]
    elapsed = [d.elapsed_time for d in diagnostics]

    # Flatten forward log-likelihoods
    all_fwd_ll = [ll for per_block in fwd_lls for ll in per_block]

    return {
        "total_mtm_steps": n,
        "accepted_count": sum(accepted),
        "accepted_ratio": sum(accepted) / n if n > 0 else 0.0,
        "mean_accept_prob": sum(accept_probs) / n if n > 0 else 0.0,
        "mean_forward_loglikelihood": (
            sum(all_fwd_ll) / len(all_fwd_ll) if all_fwd_ll else 0.0
        ),
        "total_elapsed_seconds": sum(elapsed),
        "method": diagnostics[0].method if diagnostics else "",
        "beta": diagnostics[0].beta if diagnostics else 0.0,
        "K": diagnostics[0].K if diagnostics else 0,
    }
