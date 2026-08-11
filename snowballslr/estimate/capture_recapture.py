"""Two-source capture-recapture (Chapman estimator).

    N_hat = ((n1+1)(n2+1)) / (m+1) - 1
    var   = ((n1+1)(n2+1)(n1-m)(n2-m)) / ((m+1)^2 (m+2))

Guard: with fewer than three overlapping records the estimate is refused
outright. An estimate built on near-zero overlap is worse than no estimate,
because it looks authoritative in a manuscript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RecallEstimate", "chapman", "lincoln_petersen"]

MIN_OVERLAP = 3


@dataclass(frozen=True, slots=True)
class RecallEstimate:
    method: str
    estimable: bool
    n_observed: int
    n_hat: float | None = None
    n_hat_ci: tuple[float, float] | None = None
    recall: float | None = None
    recall_ci: tuple[float, float] | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def recall_lower(self) -> float | None:
        return self.recall_ci[0] if self.recall_ci else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "estimable": self.estimable,
            "n_observed": self.n_observed,
            "n_hat": self.n_hat,
            "n_hat_ci": list(self.n_hat_ci) if self.n_hat_ci else None,
            "recall": self.recall,
            "recall_ci": list(self.recall_ci) if self.recall_ci else None,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "detail": self.detail,
        }


def lincoln_petersen(n1: int, n2: int, m: int) -> float | None:
    """Naive estimator, provided for comparison only. Chapman is preferred."""
    if m <= 0:
        return None
    return (n1 * n2) / m


def chapman(
    n1: int, n2: int, m: int, *, n_observed: int | None = None, z: float = 1.96
) -> RecallEstimate:
    """Chapman-corrected two-source estimate with Seber variance and 95% CI."""
    observed = n_observed if n_observed is not None else n1 + n2 - m

    if m < MIN_OVERLAP:
        return RecallEstimate(
            method="chapman",
            estimable=False,
            n_observed=observed,
            reason=(
                f"overlap m={m} below the minimum of {MIN_OVERLAP}; "
                "a capture-recapture estimate from near-zero overlap is not "
                "interpretable"
            ),
            detail={"n1": n1, "n2": n2, "m": m},
        )
    if n1 <= 0 or n2 <= 0:
        return RecallEstimate(
            method="chapman",
            estimable=False,
            n_observed=observed,
            reason="at least one source arm is empty",
            detail={"n1": n1, "n2": n2, "m": m},
        )

    n_hat = ((n1 + 1) * (n2 + 1)) / (m + 1) - 1
    var = (
        (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)
    ) / (((m + 1) ** 2) * (m + 2))
    se = math.sqrt(max(var, 0.0))

    lo = max(float(observed), n_hat - z * se)
    hi = n_hat + z * se

    recall = min(1.0, observed / n_hat) if n_hat > 0 else None
    recall_ci = (
        (min(1.0, observed / hi) if hi > 0 else 0.0,
         min(1.0, observed / lo) if lo > 0 else 1.0)
    )

    warnings: list[str] = []
    if n_hat < observed:
        warnings.append(
            "point estimate falls below the observed count; treat recall as 1.0 "
            "and the estimate as uninformative"
        )
    if se > n_hat:
        warnings.append("standard error exceeds the point estimate; interval is very wide")

    return RecallEstimate(
        method="chapman",
        estimable=True,
        n_observed=observed,
        n_hat=n_hat,
        n_hat_ci=(lo, hi),
        recall=recall,
        recall_ci=recall_ci,
        warnings=tuple(warnings),
        detail={
            "n1": n1,
            "n2": n2,
            "m": m,
            "se": se,
            "lincoln_petersen": lincoln_petersen(n1, n2, m),
        },
    )
