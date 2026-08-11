"""Rule composition and construction from configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import StoppingSettings
from ..core.iteration import IterationHistory
from ..errors import ConfigError
from .asymptote import AsymptoticCoverage
from .base import StopDecision
from .budget import Budget
from .exhaustion import Exhaustion
from .recall_rule import EstimatedRecall
from .yield_rule import MarginalYield

__all__ = ["RuleSet", "build_rules"]

_REGISTRY = {
    "marginal_yield": MarginalYield,
    "asymptotic_coverage": AsymptoticCoverage,
    "estimated_recall": EstimatedRecall,
    "budget": Budget,
    "exhaustion": Exhaustion,
}


def build_rules(settings: StoppingSettings) -> list[Any]:
    rules: list[Any] = []
    for spec in settings.rules:
        data = spec.model_dump()
        kind = data.pop("type")
        cls = _REGISTRY.get(kind)
        if cls is None:
            raise ConfigError(
                f"unknown stopping rule '{kind}'; available: {sorted(_REGISTRY)}"
            )
        rules.append(cls(**data))
    if not any(isinstance(r, Exhaustion) for r in rules):
        rules.append(Exhaustion())
    return rules


class RuleSet:
    """Evaluates every rule each iteration, then combines the decisions."""

    def __init__(self, rules: Sequence[Any], mode: str = "any_of") -> None:
        if mode not in ("any_of", "all_of"):
            raise ConfigError(f"stopping.mode must be any_of or all_of, got {mode}")
        self.rules = list(rules)
        self.mode = mode

    def evaluate(self, history: IterationHistory) -> tuple[bool, list[StopDecision]]:
        decisions = [r.evaluate(history) for r in self.rules]
        fired = [d for d in decisions if d.triggered]
        # Exhaustion always terminates: there is nothing left to expand.
        if any(d.rule == "exhaustion" and d.triggered for d in decisions):
            return True, decisions
        if self.mode == "any_of":
            return bool(fired), decisions
        applicable = [d for d in decisions if d.rule != "exhaustion"]
        return bool(applicable) and all(d.triggered for d in applicable), decisions

    def stopped_by(self, decisions: Sequence[StopDecision]) -> str | None:
        fired = [d.rule for d in decisions if d.triggered]
        return fired[0] if fired else None
