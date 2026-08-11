"""Exhaustion rule: the frontier is empty.

Always active. Doubles as the upper-bound baseline in the validation harness.
"""

from __future__ import annotations

from ..core.iteration import IterationHistory
from .base import StopDecision

__all__ = ["Exhaustion"]


class Exhaustion:
    name = "exhaustion"

    def evaluate(self, history: IterationHistory) -> StopDecision:
        last = history.last
        if last is None:
            return StopDecision(self.name, False, None, None, "No iterations yet.", {})
        triggered = last.frontier_size == 0
        rationale = (
            f"Citation searching was continued until the frontier was exhausted after "
            f"iteration {last.n}: no newly included study remained whose references and "
            f"citations had not already been retrieved."
            if triggered
            else f"{last.frontier_size} studies remain to be expanded."
        )
        return StopDecision(
            self.name, triggered, float(last.frontier_size), 0.0, rationale, {}
        )
