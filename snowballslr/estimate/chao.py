"""Chao1 lower bound from capture frequencies.

    N_hat = S_obs + f1^2 / (2*f2)          (f2 > 0)
    N_hat = S_obs + f1*(f1-1)/2            (f2 = 0, bias-corrected)

Distribution-free lower bound, and the direct point of comparison with Bron et
al. (2025), who used Chao's estimator as a stopping criterion on the screening
side rather than on the citation graph.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from .capture_recapture import RecallEstimate

__all__ = ["chao1", "frequency_counts"]


def frequency_counts(capture_histories: Mapping[str, Sequence[str]]) -> Counter[int]:
    """Map each record to the number of arms that captured it."""
    return Counter(len(set(arms)) for arms in capture_histories.values())


def chao1(capture_histories: Mapping[str, Sequence[str]]) -> RecallEstimate:
    counts = frequency_counts(capture_histories)
    s_obs = sum(counts.values())
    f1 = counts.get(1, 0)
    f2 = counts.get(2, 0)

    if s_obs == 0:
        return RecallEstimate(
            method="chao1",
            estimable=False,
            n_observed=0,
            reason="no observed records",
        )
    if f1 == 0:
        return RecallEstimate(
            method="chao1",
            estimable=True,
            n_observed=s_obs,
            n_hat=float(s_obs),
            recall=1.0,
            recall_ci=(1.0, 1.0),
            warnings=("no singletons observed; the lower bound coincides with S_obs",),
            detail={"f1": f1, "f2": f2, "s_obs": s_obs},
        )

    if f2 == 0 and f1 == s_obs:
        # Every record was seen exactly once, so the data carry no information
        # about how much was missed: the bias-corrected form degenerates to
        # S_obs + f1(f1-1)/2, which grows quadratically in the sample size. On a
        # real 478-record run with disjoint arms this returned N_hat = 114,481
        # and an implied recall of 0.4% while reporting estimable=True -- exactly
        # the authoritative-looking-but-empty number the Chapman guard refuses.
        return RecallEstimate(
            method="chao1",
            estimable=False,
            n_observed=s_obs,
            reason=(
                f"every one of the {s_obs} records was captured by exactly one arm "
                "(no doubletons); Chao1 degenerates and its estimate would be an "
                "artefact of sample size rather than of coverage. Configure arms "
                "that can capture the same record -- see the estimate.arms warning"
            ),
            detail={"f1": f1, "f2": f2, "s_obs": s_obs},
        )

    if f2 > 0:
        n_hat = s_obs + (f1**2) / (2 * f2)
        var = f2 * (
            0.5 * (f1 / f2) ** 2 + (f1 / f2) ** 3 + 0.25 * (f1 / f2) ** 4
        )
    else:
        n_hat = s_obs + f1 * (f1 - 1) / 2
        var = 0.5 * f1 * (f1 - 1) + 0.25 * f1 * (2 * f1 - 1) ** 2 - (f1**4) / (4 * n_hat)

    se = math.sqrt(max(var, 0.0))
    # Log-transformed CI (Chao 1987) keeps the lower bound above S_obs.
    if n_hat > s_obs and se > 0:
        c = math.exp(1.96 * math.sqrt(math.log(1 + var / ((n_hat - s_obs) ** 2))))
        lo = s_obs + (n_hat - s_obs) / c
        hi = s_obs + (n_hat - s_obs) * c
    else:
        lo, hi = float(s_obs), n_hat + 1.96 * se

    recall = min(1.0, s_obs / n_hat) if n_hat > 0 else None
    return RecallEstimate(
        method="chao1",
        estimable=True,
        n_observed=s_obs,
        n_hat=n_hat,
        n_hat_ci=(lo, hi),
        recall=recall,
        recall_ci=(min(1.0, s_obs / hi) if hi > 0 else 0.0, min(1.0, s_obs / lo) if lo > 0 else 1.0),
        warnings=(
            "Chao1 is a lower bound on N; the implied recall is therefore an "
            "upper bound",
        ),
        detail={"f1": f1, "f2": f2, "s_obs": s_obs, "se": se},
    )
