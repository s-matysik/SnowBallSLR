"""Marginal-yield stopping rule."""

from __future__ import annotations

from ..core.iteration import IterationHistory
from .base import StopDecision

__all__ = ["MarginalYield"]


class MarginalYield:
    """Fire when the include rate stays below ``eps`` for ``k`` consecutive rounds."""

    name = "marginal_yield"

    def __init__(self, eps: float = 0.01, k: int = 2) -> None:
        self.eps = eps
        self.k = k

    def evaluate(self, history: IterationHistory) -> StopDecision:
        if len(history) < self.k:
            return StopDecision(
                self.name,
                False,
                None,
                self.eps,
                f"Fewer than {self.k} iterations completed; rule not yet applicable.",
                {"iterations": len(history)},
            )
        window = history.stats[-self.k :]
        rates = [s.yield_rate for s in window]
        triggered = all(r < self.eps for r in rates)
        latest = rates[-1]
        if triggered:
            rationale = (
                f"Citation searching was stopped after iteration {window[-1].n}: the "
                f"proportion of newly included records fell below {self.eps:.1%} of "
                f"records screened in {self.k} consecutive iterations "
                f"(observed: {', '.join(f'{r:.2%}' for r in rates)})."
            )
        else:
            rationale = (
                f"Marginal yield in the last {self.k} iterations was "
                f"{', '.join(f'{r:.2%}' for r in rates)}, not yet below the "
                f"{self.eps:.1%} threshold."
            )
        return StopDecision(
            self.name, triggered, latest, self.eps, rationale, {"window_rates": rates}
        )
