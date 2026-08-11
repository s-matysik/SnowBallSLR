"""Stopping-rule protocol.

Every configured rule is evaluated on every iteration and *all* decisions are
recorded, including non-firing ones. This costs nothing at runtime and yields
the full rule-comparison dataset for methodological analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.iteration import IterationHistory

__all__ = ["StopDecision", "StoppingRule"]


@dataclass(frozen=True, slots=True)
class StopDecision:
    rule: str
    triggered: bool
    value: float | None = None
    threshold: float | None = None
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "triggered": self.triggered,
            "value": self.value,
            "threshold": self.threshold,
            "rationale": self.rationale,
            "detail": self.detail,
        }


@runtime_checkable
class StoppingRule(Protocol):
    name: str

    def evaluate(self, history: IterationHistory) -> StopDecision: ...
