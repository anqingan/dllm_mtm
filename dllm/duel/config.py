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
    alpha: float = 4.0  # target power exponent for p_target,DUEL(x)^alpha
    beta: float | None = None  # backward-compatible alias for alpha
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
    target_temperature: float = 1.0  # temperature for p_target,DUEL
    proposal_temperature: float | None = None  # temperature for q_Tprop
    temperature: float | None = None  # backward-compatible alias for proposal_temperature
    proposal_top_k: int | None = None
    proposal_top_p: float | None = None
    top_k: int | None = None  # backward-compatible alias for proposal_top_k
    top_p: float | None = None  # backward-compatible alias for proposal_top_p
    exact_mtm: bool = True

    def __post_init__(self):
        if self.beta is not None:
            self.alpha = self.beta
        self.beta = self.alpha

        if self.proposal_temperature is None:
            self.proposal_temperature = (
                self.temperature if self.temperature is not None else 1.0
            )
        self.temperature = self.proposal_temperature

        if self.proposal_top_k is None:
            self.proposal_top_k = self.top_k
        self.top_k = self.proposal_top_k
        if self.proposal_top_p is None:
            self.proposal_top_p = self.top_p
        self.top_p = self.proposal_top_p

        if self.method not in ("duel_mtm", "duel_rerank"):
            raise ValueError(f"Unknown method: {self.method!r}")
        if self.exact_mtm and self.method != "duel_mtm":
            raise ValueError(
                "exact_mtm=True only supports method='duel_mtm'. "
                "duel_rerank is a heuristic reranker, not an exact MTM kernel."
            )
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
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if self.proposal_temperature is None or self.proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be positive")
        if self.exact_mtm and self.rollback_policy == "uniform_step":
            raise ValueError(
                "exact_mtm=True requires deterministic rollback. Random rollback "
                "is part of q and must be included before enabling uniform_step."
            )
