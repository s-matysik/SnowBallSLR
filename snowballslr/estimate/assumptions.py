"""Capture-recapture assumption diagnostics (spec 8.3).

Every violation is reported, never silently absorbed. The direction of the bias
is stated explicitly, because the honest headline for snowballing is that source
arms are positively dependent -- which makes N_hat too small and the implied
recall an *upper* bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import Work

__all__ = ["AssumptionReport", "diagnose", "odds_ratio"]


@dataclass(frozen=True, slots=True)
class AssumptionReport:
    warnings: tuple[str, ...]
    dependence: dict[str, Any] = field(default_factory=dict)
    heterogeneity: dict[str, Any] = field(default_factory=dict)
    closure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": list(self.warnings),
            "dependence": self.dependence,
            "heterogeneity": self.heterogeneity,
            "closure": self.closure,
        }


def odds_ratio(n11: int, n10: int, n01: int, n00: int) -> float | None:
    """Haldane-Anscombe corrected odds ratio for the 2x2 capture table."""
    a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    if b * c == 0:
        return None
    return (a * d) / (b * c)


def _chi2_pvalue(chi2: float, dof: int = 1) -> float:
    """Survival function of chi-square with 1 dof, via the error function."""
    if dof != 1:
        raise ValueError("only dof=1 supported")
    return math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))


def diagnose(
    capture_histories: Mapping[str, Sequence[str]],
    works: Mapping[str, Work],
    arms: Sequence[str],
    *,
    n_hat: float | None = None,
    cache_window: tuple[str, str] | None = None,
) -> AssumptionReport:
    warnings: list[str] = []

    # -- 1. dependence between arms --------------------------------------
    #
    # With exactly two arms, independence is NOT testable: the 2x2 table has
    # one unobserved cell and zero residual degrees of freedom. Estimating that
    # cell from N_hat and then testing the table would be circular, because
    # N_hat already assumes independence. We therefore state the assumption and
    # the direction of its likely violation instead of manufacturing a p-value.
    #
    # With three or more arms, dependence between a pair IS testable on
    # observed data alone: restrict to records captured by at least one *other*
    # arm, and the resulting 2x2 table is fully observed.
    dependence: dict[str, Any] = {"n_arms": len(arms), "testable": len(arms) >= 3}

    if len(arms) == 2:
        dependence["pairs"] = []
        warnings.append(
            "Independence of the two source arms is assumed but NOT testable "
            "with two arms (zero residual degrees of freedom). In citation "
            "searching the arms are typically positively dependent, because "
            "backward and forward chasing both concentrate on well-connected "
            "literature; under positive dependence N_hat is biased downward and "
            "the estimated recall must be read as an UPPER BOUND. Configure a "
            "third source arm to test this assumption."
        )
    elif len(arms) >= 3:
        pairs: list[dict[str, Any]] = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a1, a2 = arms[i], arms[j]
                others = [a for k, a in enumerate(arms) if k not in (i, j)]
                subset = [
                    set(h)
                    for h in capture_histories.values()
                    if any(o in h for o in others)
                ]
                if len(subset) < 10:
                    continue
                n11 = sum(1 for h in subset if a1 in h and a2 in h)
                n10 = sum(1 for h in subset if a1 in h and a2 not in h)
                n01 = sum(1 for h in subset if a1 not in h and a2 in h)
                n00 = sum(1 for h in subset if a1 not in h and a2 not in h)
                or_ = odds_ratio(n11, n10, n01, n00)
                total = n11 + n10 + n01 + n00
                chi2 = 0.0
                rows = ((n11 + n10), (n01 + n00), (n11 + n01), (n10 + n00))
                if total > 0 and all(rows):
                    chi2 = (
                        total
                        * max(0.0, abs(n11 * n00 - n10 * n01) - total / 2) ** 2
                        / (rows[0] * rows[1] * rows[2] * rows[3])
                    )
                p = _chi2_pvalue(chi2)
                pairs.append(
                    {
                        "arms": [a1, a2],
                        "conditioned_on": others,
                        "table": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
                        "odds_ratio": or_,
                        "chi2": chi2,
                        "p_value": p,
                    }
                )
                if or_ is not None and or_ > 1.2 and p < 0.05:
                    warnings.append(
                        f"Source arms '{a1}' and '{a2}' are positively dependent "
                        f"(conditional OR={or_:.2f}, p={p:.3g}). N_hat is biased "
                        "downward and the estimated recall is an UPPER BOUND."
                    )
                elif or_ is not None and or_ < 0.83 and p < 0.05:
                    warnings.append(
                        f"Source arms '{a1}' and '{a2}' are negatively dependent "
                        f"(conditional OR={or_:.2f}, p={p:.3g}). N_hat is biased "
                        "upward and recall is understated."
                    )
        dependence["pairs"] = pairs

    # -- 2. heterogeneous catchability -----------------------------------
    strata: dict[str, dict[str, float]] = {}
    years = [w.year for w in works.values() if w.year]
    if years:
        median_year = sorted(years)[len(years) // 2]
        buckets: dict[str, list[int]] = {"older": [], "recent": []}
        for key, w in works.items():
            if w.year is None:
                continue
            bucket = "older" if w.year < median_year else "recent"
            buckets[bucket].append(len(set(capture_histories.get(key, ()))))
        strata["publication_year"] = {
            b: (sum(v) / len(v) if v else 0.0) for b, v in buckets.items()
        }

    lang_buckets: dict[str, list[int]] = {}
    for key, w in works.items():
        lang = (w.language or "unknown").lower()
        bucket = "en" if lang.startswith("en") else ("unknown" if lang == "unknown" else "non-en")
        lang_buckets.setdefault(bucket, []).append(len(set(capture_histories.get(key, ()))))
    if len(lang_buckets) > 1:
        strata["language"] = {
            b: (sum(v) / len(v) if v else 0.0) for b, v in sorted(lang_buckets.items())
        }

    type_buckets: dict[str, list[int]] = {}
    for key, w in works.items():
        type_buckets.setdefault(w.type or "unknown", []).append(
            len(set(capture_histories.get(key, ())))
        )
    if len(type_buckets) > 1:
        strata["venue_type"] = {
            b: (sum(v) / len(v) if v else 0.0) for b, v in sorted(type_buckets.items())
        }

    heterogeneity = {"mean_arms_per_stratum": strata}
    for dimension, values in strata.items():
        if len(values) < 2:
            continue
        lo_name = min(values, key=lambda k: values[k])
        hi_name = max(values, key=lambda k: values[k])
        lo, hi = values[lo_name], values[hi_name]
        if hi > 0 and (hi - lo) / hi > 0.25:
            warnings.append(
                f"Catchability is heterogeneous across {dimension}: '{lo_name}' "
                f"records are captured by {lo:.2f} arms on average versus "
                f"{hi:.2f} for '{hi_name}'. N_hat is biased downward."
            )

    # -- 3. closure ------------------------------------------------------
    closure: dict[str, Any] = {}
    if cache_window:
        closure = {"first_request": cache_window[0], "last_request": cache_window[1]}
        if cache_window[0][:10] != cache_window[1][:10]:
            warnings.append(
                "Requests span more than one calendar day "
                f"({cache_window[0][:10]} to {cache_window[1][:10]}); the "
                "underlying population was not closed during data collection."
            )

    unresolved = sum(1 for w in works.values() if w.unresolved)
    if unresolved:
        closure["unresolved_records"] = unresolved
        warnings.append(
            f"{unresolved} discovered records could not be resolved to a "
            "persistent identifier and are excluded from the capture table."
        )

    return AssumptionReport(
        warnings=tuple(warnings),
        dependence=dependence,
        heterogeneity=heterogeneity,
        closure=closure,
    )
