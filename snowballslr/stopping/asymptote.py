"""Asymptotic-coverage stopping rule.

Fits ``N(i) = N_inf * (1 - exp(-lambda * i))`` to the cumulative-includes curve
and fires when the observed count reaches ``tau`` of the fitted asymptote. The
initial guess and iteration bound are fixed, so the fit is reproducible.
"""

from __future__ import annotations

import math

from ..core.iteration import IterationHistory
from .base import StopDecision

__all__ = ["AsymptoticCoverage"]


def _model(i, n_inf, lam):
    import numpy as np

    return n_inf * (1.0 - np.exp(-lam * i))


class AsymptoticCoverage:
    name = "asymptotic_coverage"

    def __init__(self, tau: float = 0.95, min_points: int = 4) -> None:
        self.tau = tau
        self.min_points = min_points

    def evaluate(self, history: IterationHistory) -> StopDecision:
        if len(history) < self.min_points:
            return StopDecision(
                self.name,
                False,
                None,
                self.tau,
                f"Fewer than {self.min_points} iterations; curve fit not attempted.",
                {"iterations": len(history)},
            )

        import numpy as np
        from scipy.optimize import curve_fit

        xs = np.array([s.n for s in history], dtype=float)
        ys = np.array(history.cumulative_included, dtype=float)
        n_obs = float(ys[-1])
        if n_obs <= 0:
            return StopDecision(
                self.name, False, None, self.tau, "No included records yet.", {}
            )

        try:
            popt, _ = curve_fit(
                _model,
                xs,
                ys,
                p0=[2.0 * n_obs, 0.5],
                bounds=([n_obs, 1e-4], [np.inf, 10.0]),
                maxfev=20000,
            )
        except Exception as exc:
            return StopDecision(
                self.name,
                False,
                None,
                self.tau,
                "Accumulation-curve fit did not converge; rule inconclusive.",
                {"fit_failed": True, "error": type(exc).__name__},
            )

        n_inf, lam = float(popt[0]), float(popt[1])
        if not math.isfinite(n_inf) or n_inf < n_obs:
            return StopDecision(
                self.name,
                False,
                None,
                self.tau,
                "Fitted asymptote is below the observed count; rule inconclusive.",
                {"fit_failed": True, "n_inf": n_inf},
            )

        coverage = n_obs / n_inf
        triggered = coverage >= self.tau
        rationale = (
            f"An exponential accumulation curve fitted to cumulative inclusions "
            f"estimates an asymptote of {n_inf:.1f} studies; the {n_obs:.0f} studies "
            f"found represent {coverage:.1%} of that asymptote"
            + (
                f", meeting the pre-specified {self.tau:.0%} coverage criterion."
                if triggered
                else f", below the pre-specified {self.tau:.0%} criterion."
            )
        )
        return StopDecision(
            self.name,
            triggered,
            coverage,
            self.tau,
            rationale,
            {"n_inf": n_inf, "lambda": lam, "n_observed": n_obs},
        )
