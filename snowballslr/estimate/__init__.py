"""Recall estimation via capture-recapture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..types import Direction, Work
from .assumptions import AssumptionReport, diagnose, odds_ratio
from .capture_recapture import RecallEstimate, chapman, lincoln_petersen
from .chao import chao1, frequency_counts
from .loglinear import loglinear

__all__ = [
    "AssumptionReport",
    "RecallEstimate",
    "capture_histories",
    "chao1",
    "chapman",
    "diagnose",
    "estimate_recall",
    "frequency_counts",
    "lincoln_petersen",
    "loglinear",
    "odds_ratio",
]


def _matches(work: Work, criteria: Mapping[str, str]) -> bool:
    for field, expected in criteria.items():
        if field == "direction":
            if Direction(expected) not in work.directions:
                return False
        elif field == "provider":
            if expected not in work.providers:
                return False
        else:
            if getattr(work, field, None) != expected:
                return False
    return True


def capture_histories(
    included: Sequence[Work], arms: Sequence[Mapping[str, object]]
) -> dict[str, list[str]]:
    """Which arms captured each included work.

    Computed over included works only: capture-recapture estimates the size of
    the *relevant* population, not of everything the crawl touched.
    """
    out: dict[str, list[str]] = {}
    for work in sorted(included, key=lambda w: w.key):
        hit = [
            str(arm["name"])
            for arm in arms
            if _matches(work, dict(arm.get("filter") or {}))  # type: ignore[arg-type]
        ]
        if hit:
            out[work.key] = sorted(set(hit))
    return out


def estimate_recall(
    included: Sequence[Work],
    arms: Sequence[Mapping[str, object]],
    *,
    method: str = "chapman",
    with_diagnostics: bool = True,
    cache_window: tuple[str, str] | None = None,
) -> RecallEstimate:
    """Estimate recall of the union of source arms."""
    histories = capture_histories(included, arms)
    arm_names = [str(a["name"]) for a in arms]

    if method == "chao":
        est = chao1(histories)
    elif method == "loglinear":
        est = loglinear(histories, arm_names)
    else:
        if len(arm_names) < 2:
            return RecallEstimate(
                method="chapman",
                estimable=False,
                n_observed=len(histories),
                reason="at least two source arms are required",
            )
        a1, a2 = arm_names[0], arm_names[1]
        n1 = sum(1 for h in histories.values() if a1 in h)
        n2 = sum(1 for h in histories.values() if a2 in h)
        m = sum(1 for h in histories.values() if a1 in h and a2 in h)
        est = chapman(n1, n2, m, n_observed=len(histories))

    if not with_diagnostics:
        return est

    works = {w.key: w for w in included}
    report = diagnose(
        histories, works, arm_names, n_hat=est.n_hat, cache_window=cache_window
    )
    return RecallEstimate(
        method=est.method,
        estimable=est.estimable,
        n_observed=est.n_observed,
        n_hat=est.n_hat,
        n_hat_ci=est.n_hat_ci,
        recall=est.recall,
        recall_ci=est.recall_ci,
        reason=est.reason,
        warnings=tuple(est.warnings) + report.warnings,
        detail={**est.detail, "assumptions": report.to_dict()},
    )
