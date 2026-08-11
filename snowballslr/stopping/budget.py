"""Budget stopping rule: screening capacity or minimum acceptable yield."""

from __future__ import annotations

from ..core.iteration import IterationHistory
from .base import StopDecision

__all__ = ["Budget"]


class Budget:
    name = "budget"

    def __init__(
        self, max_screened: int | None = None, min_yield_per_100: float | None = None
    ) -> None:
        self.max_screened = max_screened
        self.min_yield_per_100 = min_yield_per_100

    def evaluate(self, history: IterationHistory) -> StopDecision:
        last = history.last
        if last is None:
            return StopDecision(self.name, False, None, None, "No iterations yet.", {})

        if self.max_screened is not None and last.cumulative_screened >= self.max_screened:
            return StopDecision(
                self.name,
                True,
                float(last.cumulative_screened),
                float(self.max_screened),
                (
                    f"Citation searching was stopped after {last.cumulative_screened} "
                    f"records had been screened, reaching the pre-specified screening "
                    f"budget of {self.max_screened}."
                ),
                {"criterion": "max_screened"},
            )

        if self.min_yield_per_100 is not None:
            per100 = last.yield_rate * 100.0
            if per100 < self.min_yield_per_100:
                return StopDecision(
                    self.name,
                    True,
                    per100,
                    self.min_yield_per_100,
                    (
                        f"Citation searching was stopped after iteration {last.n}: the "
                        f"yield of {per100:.2f} included studies per 100 screened fell "
                        f"below the pre-specified minimum of {self.min_yield_per_100}."
                    ),
                    {"criterion": "min_yield_per_100"},
                )
            return StopDecision(
                self.name,
                False,
                per100,
                self.min_yield_per_100,
                f"Yield of {per100:.2f} per 100 screened is above the minimum.",
                {"criterion": "min_yield_per_100"},
            )

        return StopDecision(
            self.name,
            False,
            float(last.cumulative_screened),
            float(self.max_screened) if self.max_screened else None,
            f"{last.cumulative_screened} records screened; budget not reached.",
            {},
        )
