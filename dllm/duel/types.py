"""Shared data structures for DUEL-guided Intra-Block MTM."""

from dataclasses import dataclass, field

import torch


@dataclass
class DuelDiagnostics:
    """Diagnostics recorded for a single MTM step."""

    block_index: int = -1
    block_start: int = -1
    block_end: int = -1
    rollback_positions: list[int] = field(default_factory=list)
    selected_index: int = -1
    forward_loglikelihoods: list[float] = field(default_factory=list)
    backward_loglikelihoods: list[float] = field(default_factory=list)
    forward_log_weights: list[float] = field(default_factory=list)
    backward_log_weights: list[float] = field(default_factory=list)
    log_W_fwd: float = float("-inf")
    log_W_bwd: float = float("-inf")
    accept_prob: float = 0.0
    accepted: bool = False
    method: str = ""
    unmask_rule: str = ""
    beta: float = 0.0
    K: int = 0
    elapsed_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "block_index": self.block_index,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "rollback_positions": self.rollback_positions,
            "selected_index": self.selected_index,
            "forward_loglikelihoods": self.forward_loglikelihoods,
            "backward_loglikelihoods": self.backward_loglikelihoods,
            "forward_log_weights": self.forward_log_weights,
            "backward_log_weights": self.backward_log_weights,
            "log_W_fwd": self.log_W_fwd,
            "log_W_bwd": self.log_W_bwd,
            "accept_prob": self.accept_prob,
            "accepted": self.accepted,
            "method": self.method,
            "unmask_rule": self.unmask_rule,
            "beta": self.beta,
            "K": self.K,
            "elapsed_time": self.elapsed_time,
        }
