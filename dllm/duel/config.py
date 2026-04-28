"""Configuration for DUEL-guided Intra-Block MTM."""

from dataclasses import dataclass, field


@dataclass
class DuelMTMConfig:
    """Configuration for DUEL-guided Intra-Block MTM decoding.

    All fields default to off/no-op so that enabled=false preserves
    the original DLLM generation behaviour exactly.
    """

    enabled: bool = False
    method: str = "duel_mtm"  # choices: "duel_mtm", "duel_rerank"
    K: int = 4  # number of forward proposals
    beta: float = 4.0  # temperature exponent for DUEL target
    block_size: int = 32  # block size for intra-block refinement
    num_mtm_steps_per_block: int = 1  # MTM iterations per block
    unmask_rule: str = "probability_margin"  # deterministic unmasking rule
    positions_per_step: int = 1  # positions revealed per DUEL step
    rollback_policy: str = "fixed_ratio"  # "fixed_ratio" or "uniform_step"
    rollback_ratio: float = 0.5  # fraction of block to re-mask
    score_scope: str = "rollback_region"  # "rollback_region" or "full_block"
    seed: int = 42
    debug: bool = False
    save_diagnostics: bool = False
    diagnostics_path: str = "duel_mtm_diagnostics.jsonl"
    temperature: float = 1.0  # token sampling temperature
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self):
        if self.method not in ("duel_mtm", "duel_rerank"):
            raise ValueError(f"Unknown method: {self.method!r}")
        if self.unmask_rule not in (
            "left_to_right",
            "greedy_confidence",
            "probability_margin",
        ):
            raise ValueError(f"Unknown unmask_rule: {self.unmask_rule!r}")
        if self.rollback_policy not in ("fixed_ratio", "uniform_step"):
            raise ValueError(f"Unknown rollback_policy: {self.rollback_policy!r}")
        if self.score_scope not in ("rollback_region", "full_block"):
            raise ValueError(f"Unknown score_scope: {self.score_scope!r}")
