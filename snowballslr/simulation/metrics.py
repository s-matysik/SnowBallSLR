"""Validation metrics.

The headline metrics are not recall and precision -- those are properties of
snowballing itself, already studied. They are the *stopping-rule* metrics:
where each rule fired relative to the oracle-optimal stop, and how well the
capture-recapture estimate tracks true recall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RecallCurve",
    "StoppingOutcome",
    "calibration",
    "evaluate_stopping_rules",
    "recall_curve",
    "screening_burden",
]


@dataclass(frozen=True, slots=True)
class RecallCurve:
    iterations: tuple[int, ...]
    recall: tuple[float, ...]
    cumulative_screened: tuple[int, ...]
    cumulative_included: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": list(self.iterations),
            "recall": list(self.recall),
            "cumulative_screened": list(self.cumulative_screened),
            "cumulative_included": list(self.cumulative_included),
        }


def recall_curve(history: Sequence[Any], n_gold: int) -> RecallCurve:
    if n_gold <= 0:
        raise ValueError("gold standard must be non-empty")
    return RecallCurve(
        iterations=tuple(s.n for s in history),
        recall=tuple(min(1.0, s.cumulative_included / n_gold) for s in history),
        cumulative_screened=tuple(s.cumulative_screened for s in history),
        cumulative_included=tuple(s.cumulative_included for s in history),
    )


def screening_burden(curve: RecallCurve, target: float) -> int | None:
    """Records screened to first reach ``target`` recall; ``None`` if never reached."""
    for r, screened in zip(curve.recall, curve.cumulative_screened, strict=False):
        if r >= target:
            return screened
    return None


@dataclass(frozen=True, slots=True)
class StoppingOutcome:
    rule: str
    fired_at_iteration: int | None
    recall_at_stop: float | None
    optimal_iteration: int
    overshoot_iterations: int | None
    excess_screened: int | None
    missed_studies: int | None
    regret: float | None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "fired_at_iteration": self.fired_at_iteration,
            "recall_at_stop": self.recall_at_stop,
            "optimal_iteration": self.optimal_iteration,
            "overshoot_iterations": self.overshoot_iterations,
            "excess_screened": self.excess_screened,
            "missed_studies": self.missed_studies,
            "regret": self.regret,
            "detail": self.detail,
        }


def evaluate_stopping_rules(
    curve: RecallCurve,
    decisions_by_iteration: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    target_recall: float = 0.95,
    n_gold: int | None = None,
) -> list[StoppingOutcome]:
    """Compare every rule against the oracle-optimal stopping point.

    The optimal stop is the first iteration at which ``target_recall`` is
    reached: stopping earlier misses studies, stopping later wastes screening.
    """
    total = n_gold or (max(curve.cumulative_included) if curve.cumulative_included else 0)
    optimal_idx = next(
        (i for i, r in enumerate(curve.recall) if r >= target_recall),
        len(curve.recall) - 1,
    )
    optimal_iteration = curve.iterations[optimal_idx] if curve.iterations else 0
    optimal_screened = (
        curve.cumulative_screened[optimal_idx] if curve.cumulative_screened else 0
    )

    rules: list[str] = sorted(
        {d["rule"] for decisions in decisions_by_iteration.values() for d in decisions}
    )
    index = {n: i for i, n in enumerate(curve.iterations)}
    outcomes: list[StoppingOutcome] = []

    for rule in rules:
        fired_at = next(
            (
                n
                for n in sorted(decisions_by_iteration)
                for d in decisions_by_iteration[n]
                if d["rule"] == rule and d.get("triggered")
            ),
            None,
        )
        if fired_at is None or fired_at not in index:
            outcomes.append(
                StoppingOutcome(
                    rule=rule,
                    fired_at_iteration=None,
                    recall_at_stop=None,
                    optimal_iteration=optimal_iteration,
                    overshoot_iterations=None,
                    excess_screened=None,
                    missed_studies=None,
                    regret=None,
                    detail={"never_fired": True},
                )
            )
            continue

        i = index[fired_at]
        recall_at_stop = curve.recall[i]
        screened_at_stop = curve.cumulative_screened[i]
        missed = max(0, round((target_recall - recall_at_stop) * total))
        excess = screened_at_stop - optimal_screened
        # Regret: wasted screening when late, missed studies when early.
        regret = (
            float(excess) / max(1, optimal_screened)
            if excess > 0
            else float(missed) / max(1, total)
        )
        outcomes.append(
            StoppingOutcome(
                rule=rule,
                fired_at_iteration=fired_at,
                recall_at_stop=recall_at_stop,
                optimal_iteration=optimal_iteration,
                overshoot_iterations=fired_at - optimal_iteration,
                excess_screened=excess,
                missed_studies=missed,
                regret=regret,
            )
        )
    return outcomes


def calibration(
    estimated: Sequence[float], actual: Sequence[float]
) -> dict[str, float | int]:
    """Bias and error of estimated recall against the truth."""
    pairs = [(e, a) for e, a in zip(estimated, actual, strict=False) if e is not None and a is not None]
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    bias = sum(e - a for e, a in pairs) / n
    mae = sum(abs(e - a) for e, a in pairs) / n
    rmse = (sum((e - a) ** 2 for e, a in pairs) / n) ** 0.5
    overestimates = sum(1 for e, a in pairs if e > a)
    return {
        "n": n,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "overestimate_rate": overestimates / n,
    }
