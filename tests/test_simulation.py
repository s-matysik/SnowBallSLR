"""Validation harness: metrics behave as documented."""

from __future__ import annotations

from snowballslr.config import Config
from snowballslr.core.iteration import IterationStats
from snowballslr.simulation import (
    ReviewSpec,
    SeedStrategy,
    calibration,
    evaluate_stopping_rules,
    recall_curve,
    run_trial,
    screening_burden,
)
from snowballslr.simulation.harness import select_seeds


def _history(rows):
    out = []
    cum_s = cum_i = 0
    for n, (screened, included) in enumerate(rows, 1):
        cum_s += screened
        cum_i += included
        out.append(
            IterationStats(n, 1, screened, screened, screened, screened, included,
                           screened - included, cum_s, cum_i, 0)
        )
    return out


def test_recall_curve_and_burden():
    curve = recall_curve(_history([(100, 5), (100, 3), (100, 2)]), n_gold=10)
    assert curve.recall == (0.5, 0.8, 1.0)
    assert screening_burden(curve, 0.8) == 200
    assert screening_burden(curve, 1.1) is None


def test_stopping_outcome_measures_overshoot_and_regret():
    curve = recall_curve(_history([(100, 8), (100, 2), (100, 0), (100, 0)]), n_gold=10)
    decisions = {
        1: [{"rule": "early", "triggered": True}],
        2: [{"rule": "optimal", "triggered": True}],
        4: [{"rule": "late", "triggered": True}],
    }
    outcomes = {o.rule: o for o in evaluate_stopping_rules(curve, decisions, target_recall=1.0)}
    assert outcomes["early"].missed_studies == 2
    assert outcomes["late"].overshoot_iterations > 0
    assert outcomes["late"].excess_screened > 0


def test_rule_that_never_fires_is_reported():
    curve = recall_curve(_history([(100, 10)]), n_gold=10)
    outcomes = evaluate_stopping_rules(curve, {1: [{"rule": "never", "triggered": False}]})
    assert outcomes[0].fired_at_iteration is None
    assert outcomes[0].detail["never_fired"]


def test_calibration_reports_bias_direction():
    result = calibration([0.95, 0.92, 0.99], [0.80, 0.85, 0.90])
    assert result["bias"] > 0
    assert result["overestimate_rate"] == 1.0


def test_seed_strategies_are_deterministic_and_distinct(graph, oracle):
    spec = ReviewSpec(name="mini", graph=graph, gold=oracle.included, domain="test")
    import random

    picks = {
        s: select_seeds(spec, s, 3, random.Random(f"mini|{s}|3|0"))
        for s in SeedStrategy.ALL
    }
    assert all(len(v) == 3 for v in picks.values())
    assert picks[SeedStrategy.MOST_CITED] != picks[SeedStrategy.OLDEST]
    again = select_seeds(spec, SeedStrategy.RANDOM, 3, random.Random("mini|random|3|0"))
    assert again == picks[SeedStrategy.RANDOM]


def test_run_trial_produces_a_complete_result(tmp_path, graph, oracle):
    spec = ReviewSpec(name="mini", graph=graph, gold=oracle.included, domain="test")
    cfg = Config.from_dict({"run": {"max_iterations": 6}})
    result = run_trial(spec, SeedStrategy.MOST_CITED, 3, 0, tmp_path, cfg)
    assert result.n_gold == len(oracle)
    assert result.final_recall >= 0.9
    assert result.stopping_outcomes
    assert result.curve["recall"]
