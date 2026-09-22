"""Log-linear capture-recapture for three or more source arms.

Fits Poisson GLMs on the 2^k - 1 observable capture-history cells, selects among
independence and pairwise-interaction models by AIC, and estimates the missing
cell. ``statsmodels`` is imported lazily so the core install stays light.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence

from ..errors import EstimationError
from .capture_recapture import RecallEstimate

__all__ = ["loglinear"]

#: Largest population estimate, as a multiple of the observed count, that a
#: log-linear fit may return before it is refused as unidentified. Calibrated in
#: supplementary Section S13; see the plausibility guard below.
_MAX_N_HAT_MULTIPLE = 3.0


def _cells(capture_histories: Mapping[str, Sequence[str]], arms: Sequence[str]):
    counts: dict[tuple[int, ...], int] = {}
    for record_arms in capture_histories.values():
        present = set(record_arms)
        pattern = tuple(1 if a in present else 0 for a in arms)
        if sum(pattern) == 0:
            continue
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def loglinear(
    capture_histories: Mapping[str, Sequence[str]], arms: Sequence[str]
) -> RecallEstimate:
    arms = list(arms)
    if len(arms) < 3:
        raise EstimationError("log-linear estimation requires at least three arms")

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise EstimationError(
            "log-linear estimation requires statsmodels and pandas"
        ) from exc

    counts = _cells(capture_histories, arms)
    s_obs = sum(counts.values())
    if s_obs == 0:
        return RecallEstimate(
            method="loglinear", estimable=False, n_observed=0, reason="no observed records"
        )

    rows = []
    for pattern in itertools.product([0, 1], repeat=len(arms)):
        if sum(pattern) == 0:
            continue
        row = {f"a{i}": pattern[i] for i in range(len(arms))}
        row["count"] = counts.get(pattern, 0)
        rows.append(row)
    df = pd.DataFrame(rows)

    main = " + ".join(f"a{i}" for i in range(len(arms)))
    candidates: list[str] = [f"count ~ {main}"]
    pairs = list(itertools.combinations(range(len(arms)), 2))
    for r in range(1, len(pairs) + 1):
        for combo in itertools.combinations(pairs, r):
            terms = " + ".join(f"a{i}:a{j}" for i, j in combo)
            candidates.append(f"count ~ {main} + {terms}")

    best = None
    for formula in candidates:
        # The search deliberately fits saturated and near-saturated designs.
        # Perfect separation and zero-residual-variance warnings are expected
        # for those, and AIC discards them anyway -- so they are not surfaced
        # to the user as if something had gone wrong.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.glm(
                    formula=formula, data=df, family=sm.families.Poisson()
                ).fit()
                if not np.isfinite(model.aic):
                    continue
                aic, bic = float(model.aic), float(model.bic)
            except Exception:  # pragma: no cover - degenerate designs
                continue
        if best is None or aic < best[1]:
            best = (formula, aic, bic, model)

    if best is None:
        return RecallEstimate(
            method="loglinear",
            estimable=False,
            n_observed=s_obs,
            reason="no log-linear model converged on this capture table",
        )

    formula, aic, bic, model = best
    zero = pd.DataFrame([{f"a{i}": 0 for i in range(len(arms))}])
    m0 = float(model.predict(zero).iloc[0])
    n_hat = s_obs + m0
    recall = min(1.0, s_obs / n_hat) if n_hat > 0 else None

    cell_detail = {"".join(map(str, k)): v for k, v in sorted(counts.items())}

    # An arm that captured nothing is not a capture occasion. Leaving it in the model
    # lets a two-arm design be reported as a three-arm one.
    empty_arms = [
        arms[i] for i in range(len(arms))
        if not any(pattern[i] and n for pattern, n in counts.items())
    ]
    if empty_arms:
        return RecallEstimate(
            method="loglinear",
            estimable=False,
            n_observed=s_obs,
            reason=(
                f"arm(s) {', '.join(empty_arms)} captured no records, so the design is not "
                f"the {len(arms)}-arm design the model assumes"
            ),
            detail={"formula": formula, "arms": arms, "cells": cell_detail,
                    "empty_arms": empty_arms},
        )

    # Identifiability guard. A selected model carrying an interaction term needs the
    # corresponding two-arm capture cell to be observed; when that cell is
    # structurally zero the missing-cell prediction is an extrapolation with no data
    # behind it, and the fit can return an arbitrarily large population. Refusing is
    # the only safe answer: a number here would be reported to a reviewer as an
    # estimate of how much literature remains unfound.
    if ":" in formula:
        empty_pairs = [
            (arms[i], arms[j])
            for i, j in itertools.combinations(range(len(arms)), 2)
            if counts.get(tuple(1 if k in (i, j) else 0 for k in range(len(arms))), 0) == 0
        ]
        if empty_pairs:
            pair = ", ".join(f"{a} and {b}" for a, b in empty_pairs[:3])
            return RecallEstimate(
                method="loglinear",
                estimable=False,
                n_observed=s_obs,
                reason=(
                    "the selected log-linear model carries an interaction term whose "
                    f"capture cell is empty ({pair}), so the missing cell is not "
                    "identified from these data"
                ),
                detail={"formula": formula, "arms": arms, "cells": cell_detail,
                        "empty_pairwise_cells": [list(p) for p in empty_pairs]},
            )

    # Plausibility ceiling. Across the 1,620 simulated reviews of the coverage study
    # (supplementary Section S13) no log-linear fit within 50% of the truth exceeded
    # 2.09 times the observed count, so a ceiling of three rejects none of the 1,497
    # accurate fits while catching unidentified blow-ups.
    if n_hat > _MAX_N_HAT_MULTIPLE * s_obs:
        return RecallEstimate(
            method="loglinear",
            estimable=False,
            n_observed=s_obs,
            reason=(
                f"the fitted population {n_hat:.3g} exceeds {_MAX_N_HAT_MULTIPLE} times the "
                f"{s_obs} records observed, which no capture table of this size supports"
            ),
            detail={"formula": formula, "arms": arms, "cells": cell_detail,
                    "n_hat_rejected": n_hat},
        )

    return RecallEstimate(
        method="loglinear",
        estimable=True,
        n_observed=s_obs,
        n_hat=n_hat,
        recall=recall,
        warnings=(
            "log-linear estimates are sensitive to model choice; the selected "
            f"model was: {formula}",
        ),
        detail={
            "formula": formula,
            "aic": aic,
            "bic": bic,
            "missing_cell": m0,
            "arms": arms,
            "cells": {"".join(map(str, k)): v for k, v in sorted(counts.items())},
        },
    )
