"""Stopping rules fire where they should on synthetic accumulation curves."""

from __future__ import annotations

from snowballslr.core.iteration import IterationHistory, IterationStats
from snowballslr.estimate.capture_recapture import RecallEstimate
from snowballslr.stopping import (
    AsymptoticCoverage,
    Budget,
    EstimatedRecall,
    Exhaustion,
    MarginalYield,
    RuleSet,
)


def build_history(rows, frontier=5):
    h = IterationHistory()
    cum_s = cum_i = 0
    for n, (screened, included) in enumerate(rows, 1):
        cum_s += screened
        cum_i += included
        h.append(
            IterationStats(
                n=n,
                n_expanded=5,
                n_raw=screened,
                n_after_dedup=screened,
                n_eligible=screened,
                n_screened=screened,
                n_included=included,
                n_excluded=screened - included,
                cumulative_screened=cum_s,
                cumulative_included=cum_i,
                frontier_size=frontier,
            )
        )
    return h


def test_marginal_yield_needs_k_consecutive_low_iterations():
    rule = MarginalYield(eps=0.01, k=2)
    assert not rule.evaluate(build_history([(500, 40)])).triggered
    assert not rule.evaluate(build_history([(500, 40), (200, 10)])).triggered
    assert rule.evaluate(build_history([(500, 40), (200, 1), (200, 0)])).triggered


def test_marginal_yield_rationale_is_methods_ready():
    d = MarginalYield(eps=0.01, k=2).evaluate(build_history([(500, 40), (200, 1), (200, 0)]))
    assert d.rationale.startswith("Citation searching was stopped")
    assert "%" in d.rationale


def test_asymptotic_coverage_fires_on_a_saturating_curve():
    d = AsymptoticCoverage(tau=0.95).evaluate(
        build_history([(500, 40), (300, 12), (200, 3), (150, 1), (120, 0)])
    )
    assert d.triggered and d.value >= 0.95


def test_asymptotic_coverage_waits_for_enough_points():
    d = AsymptoticCoverage(min_points=4).evaluate(build_history([(500, 40), (300, 12)]))
    assert not d.triggered


def test_budget_fires_on_screening_cap():
    d = Budget(max_screened=600).evaluate(build_history([(500, 40), (200, 5)]))
    assert d.triggered and "budget" in d.rationale


def test_budget_fires_on_low_yield():
    d = Budget(min_yield_per_100=2.0).evaluate(build_history([(500, 40), (200, 1)]))
    assert d.triggered


def test_exhaustion_fires_only_on_empty_frontier():
    assert not Exhaustion().evaluate(build_history([(100, 5)], frontier=3)).triggered
    assert Exhaustion().evaluate(build_history([(100, 5)], frontier=0)).triggered


def _settled(h, n_hat=21.0, recall=0.96, recall_ci=(0.90, 1.0), n_observed=20):
    """Attach an estimate plus a flat N-hat trail, so the closure guard is satisfied."""
    est = RecallEstimate(
        method="chapman",
        estimable=True,
        n_observed=n_observed,
        n_hat=n_hat,
        recall=recall,
        recall_ci=recall_ci,
    )
    for n in (1, 2, 3):
        h.record_estimate(n, est)
    return h


def test_estimated_recall_uses_lower_ci_by_default():
    h = _settled(build_history([(100, 20)]))
    assert not EstimatedRecall(tau=0.95).evaluate(h).triggered
    assert EstimatedRecall(tau=0.95, use_lower_ci=False).evaluate(h).triggered


def test_estimated_recall_is_withheld_while_the_population_is_still_growing():
    """Closure guard: N-hat still moving means the estimate is premature (E4)."""
    h = build_history([(100, 20)])
    for n, n_hat in ((1, 20.0), (2, 40.0), (3, 90.0)):
        h.record_estimate(
            n,
            RecallEstimate(
                method="chapman", estimable=True, n_observed=20,
                n_hat=n_hat, recall=0.99, recall_ci=(0.98, 1.0),
            ),
        )
    d = EstimatedRecall(tau=0.95).evaluate(h)
    assert not d.triggered
    assert d.detail["threshold_met"] and not d.detail["closure_stable"]
    assert "still moving" in d.rationale


def test_estimated_recall_fires_once_the_population_has_settled():
    h = _settled(build_history([(100, 20)]), recall_ci=(0.97, 1.0))
    d = EstimatedRecall(tau=0.95).evaluate(h)
    assert d.triggered and d.detail["closure_stable"]


def test_closure_guard_can_be_disabled():
    h = build_history([(100, 20)])
    h.record_estimate(
        1,
        RecallEstimate(
            method="chapman", estimable=True, n_observed=20,
            n_hat=20.5, recall=0.99, recall_ci=(0.98, 1.0),
        ),
    )
    assert not EstimatedRecall(tau=0.95).evaluate(h).triggered
    assert EstimatedRecall(tau=0.95, min_stable_iterations=0).evaluate(h).triggered


def test_estimated_recall_reports_inestimable_reason():
    h = build_history([(100, 20)])
    h.recall_estimate = RecallEstimate(
        method="chapman", estimable=False, n_observed=20, reason="overlap too small"
    )
    d = EstimatedRecall().evaluate(h)
    assert not d.triggered and "overlap too small" in d.rationale


def test_ruleset_records_every_decision_not_only_the_firing_one():
    rules = RuleSet([MarginalYield(0.01, 2), Budget(max_screened=10**6), Exhaustion()])
    stop, decisions = rules.evaluate(build_history([(500, 40), (200, 1), (200, 0)]))
    assert stop
    assert len(decisions) == 3
    assert rules.stopped_by(decisions) == "marginal_yield"


def test_exhaustion_always_terminates_even_in_all_of_mode():
    rules = RuleSet([MarginalYield(0.0001, 5), Exhaustion()], mode="all_of")
    stop, _ = rules.evaluate(build_history([(100, 50)], frontier=0))
    assert stop
