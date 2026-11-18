"""Scalar evidence attached to a candidate by an existing measurement family."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One measured value or the reason it is missing.

    window_s holds causal (start, end) timestamps in the producer's clock.
    cost_ms describes the cost of this observation. Both are optional when
    callers have only images or a shared batch cost.
    """

    family: str
    version: str
    value: float | None = None
    missing_reason: str | None = None
    window_s: tuple[float, float] | None = None
    cost_ms: float | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.missing_reason is None):
            raise ValueError("Supply either a measured value or a missing reason")

    @property
    def valid(self) -> bool:
        return self.value is not None

    def write_fields(self, candidate: dict[str, Any], prefix: str, metric: str) -> None:
        """Update the existing flat candidate fields supplied by this observation."""
        if self.valid:
            candidate[f"{prefix}_{metric}"] = self.value
        candidate[f"{prefix}_valid"] = float(self.valid)
        if self.missing_reason is not None:
            candidate[f"{prefix}_missing"] = self.missing_reason
        candidate[f"{prefix}_version"] = self.version
