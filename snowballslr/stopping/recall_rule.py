"""Capture-recapture based stopping rule.

Defaults to the *lower* bound of the 95% interval. Under positive dependence
between source arms -- the normal situation in snowballing -- the point estimate
of recall is optimistic, so stopping on it would stop early.

**Stability guard (``min_stable_iterations``).** The estimators used here assume
a *closed* population, and that assumption holds: the estimand is the fixed set
of eligible records existing at the search cutoff, and nothing enters or leaves
it during a run. What a run does violate is the separate assumption that every
member of the population can be captured on each occasion. At iteration *i* only
records within *i* citation hops of the start set are reachable at all, so the
rest have capture probability zero; the reachable frame grows every time an
included record is added to the frontier, and an estimate computed at iteration
*i* therefore sizes the frame reachable *so far*, not the population that will
be reachable at saturation. Early in a run the arms
therefore agree almost perfectly on a small reachable set, N-hat collapses onto
the observed count, and estimated recall approaches 1.0 while true recall is
still low.

Benchmarked on 270 simulated reviews with known ground truth, the unguarded rule
fired in 270/270 trials at a mean *true* recall of 0.45. Requiring N-hat to be
stable within ``max_drift`` for ``min_stable_iterations`` consecutive iterations
before the rule may fire raised mean true recall at stop to 0.906 against an
achievable ceiling of 0.913 -- i.e. it removed essentially all of the shortfall.
The guard is on by default because the failure it prevents is silent and
produces a confidently wrong number in a manuscript.
"""

from __future__ import annotations

from ..core.iteration import IterationHistory
from .base import StopDecision

__all__ = ["EstimatedRecall"]


class EstimatedRecall:
    name = "estimated_recall"

    def __init__(
        self,
        tau: float = 0.95,
        use_lower_ci: bool = True,
        method: str = "chapman",
        min_stable_iterations: int = 2,
        max_drift: float = 0.05,
        authoritative: bool = False,
    ) -> None:
        self.tau = tau
        self.use_lower_ci = use_lower_ci
        self.method = method
        self.min_stable_iterations = min_stable_iterations
        self.max_drift = max_drift
        # The estimate is a diagnostic by default: a coverage study over 1,620
        # evaluations with known truth found the estimators systematically low, and
        # their nominal intervals badly calibrated -- 0% (Chapman) and 14% (Chao1)
        # coverage over the 1,350 evaluations at depths 2-6 where all three are
        # comparable (supplementary Section S13) -- so
        # a reading at or above ``tau`` is not sufficient evidence to end a review.
        # Advisory decisions are still evaluated and recorded every iteration --
        # they simply do not terminate the run. Set ``authoritative=True`` to opt
        # into recall-based termination.
        self.authoritative = authoritative

    @property
    def advisory(self) -> bool:
        """True when this rule reports but does not terminate a run."""
        return not self.authoritative

    def _stability_status(self, history: IterationHistory) -> tuple[bool, str, dict]:
        """Has N-hat settled enough for the reachable frame to be treated as complete?

        Returns ``(stable, explanation, detail)``. Requires ``min_stable_iterations``
        consecutive iterations in which the relative change in N-hat stays within
        ``max_drift``.
        """
        if self.min_stable_iterations <= 0:
            return True, "stability guard disabled", {"guard": "disabled"}

        trail = [
            e.n_hat
            for e in history.recall_estimate_trail
            if e is not None and e.estimable and e.n_hat
        ]
        needed = self.min_stable_iterations + 1
        if len(trail) < needed:
            return (
                False,
                (
                    f"only {len(trail)} estimable iteration(s) so far; the stability "
                    f"guard requires {needed} to judge whether the estimated "
                    "population has settled"
                ),
                {"n_estimable_iterations": len(trail), "required": needed},
            )

        window = trail[-needed:]
        drifts = [
            abs(window[i + 1] - window[i]) / window[i] if window[i] else 1.0
            for i in range(len(window) - 1)
        ]
        worst = max(drifts)
        if worst > self.max_drift:
            return (
                False,
                (
                    f"the estimated population is still moving ({worst:.1%} change "
                    f"over the last {self.min_stable_iterations} iteration(s), "
                    f"tolerance {self.max_drift:.0%}); capture-recapture assumes a "
                    "closed population, and a still-growing frontier makes an "
                    "early estimate optimistic"
                ),
                {"n_hat_trail": window, "max_drift_observed": worst},
            )
        return (
            True,
            f"estimated population stable within {self.max_drift:.0%}",
            {"n_hat_trail": window, "max_drift_observed": worst},
        )

    def evaluate(self, history: IterationHistory) -> StopDecision:
        est = history.recall_estimate
        if est is None:
            return StopDecision(
                self.name,
                False,
                None,
                self.tau,
                "No recall estimate available for this iteration.",
                {},
            )
        if not est.estimable:
            return StopDecision(
                self.name,
                False,
                None,
                self.tau,
                f"Recall not estimable: {est.reason}",
                {"reason": est.reason},
            )

        value = est.recall_lower if self.use_lower_ci else est.recall
        if value is None:
            value = est.recall
        if value is None:
            return StopDecision(
                self.name, False, None, self.tau, "Recall estimate incomplete.", {}
            )

        threshold_met = value >= self.tau
        stable, stability_note, stability_detail = self._stability_status(history)
        triggered = threshold_met and stable

        bound = "lower bound of the 95% confidence interval" if self.use_lower_ci else "point estimate"
        rationale = (
            f"Capture-recapture ({est.method}) across the configured source arms "
            f"estimates a relevant population of {est.n_hat:.1f} studies, of which "
            f"{est.n_observed} were identified; the {bound} of estimated recall is "
            f"{value:.1%}"
        )
        if triggered:
            rationale += (
                f", meeting the pre-specified stopping threshold of {self.tau:.0%} "
                f"({stability_note})."
            )
        elif threshold_met and not stable:
            rationale += (
                f". The threshold of {self.tau:.0%} is met, but the rule is withheld "
                f"because {stability_note}."
            )
        else:
            rationale += f", below the pre-specified threshold of {self.tau:.0%}."

        if self.advisory:
            rationale += (
                " This rule is advisory: the estimate is reported as a diagnostic and "
                "does not terminate the run."
            )

        return StopDecision(
            self.name,
            triggered,
            value,
            self.tau,
            rationale,
            {
                "estimate": est.to_dict(),
                "used_lower_ci": self.use_lower_ci,
                "threshold_met": threshold_met,
                "closure_stable": stable,
                "closure": stability_detail,
                "advisory": self.advisory,
            },
        )
