"""Capture-recapture: correctness, guards, and the documented bias direction."""

from __future__ import annotations

import random

import pytest

from snowballslr.estimate import capture_histories, estimate_recall
from snowballslr.estimate.assumptions import odds_ratio
from snowballslr.estimate.capture_recapture import MIN_OVERLAP, chapman
from snowballslr.estimate.chao import chao1
from snowballslr.types import Direction, Provenance
from tests.conftest import make_work

ARMS = [
    {"name": "backward", "filter": {"direction": "backward"}},
    {"name": "forward", "filter": {"direction": "forward"}},
]


def _prov(direction: Direction) -> Provenance:
    return Provenance(
        provider="offline",
        direction=direction,
        parent_key="p",
        iteration=1,
        retrieved_at="2026-01-01T00:00:00Z",
        response_hash="sha256:x",
    )


def _population(n_total: int, p1: float, p2: float, rho: float, seed: int):
    """Sample a population where the two arms may be positively dependent."""
    rng = random.Random(seed)
    works = []
    for i in range(n_total):
        easy = rng.random() < rho
        in1 = rng.random() < (p1 + (1 - p1) * 0.6 if easy else p1 * 0.6)
        in2 = rng.random() < (p2 + (1 - p2) * 0.6 if easy else p2 * 0.6)
        if not (in1 or in2):
            continue
        w = make_work(f"Study {i}", 2010 + i % 15, f"A{i % 30}", f"10.1000/s{i:04d}")
        provs = []
        if in1:
            provs.append(_prov(Direction.BACKWARD))
        if in2:
            provs.append(_prov(Direction.FORWARD))
        works.append(w.with_provenance(provs))
    return works


def test_chapman_matches_hand_computation():
    est = chapman(60, 50, 30)
    expected = ((60 + 1) * (50 + 1)) / (30 + 1) - 1
    assert est.estimable
    assert est.n_hat == pytest.approx(expected)
    assert est.n_hat_ci[0] < est.n_hat < est.n_hat_ci[1]


def test_chapman_refuses_near_zero_overlap():
    est = chapman(10, 10, MIN_OVERLAP - 1)
    assert not est.estimable
    assert "overlap" in est.reason


def test_chapman_refuses_empty_arm():
    assert not chapman(0, 10, 0).estimable


def test_chapman_recovers_population_when_arms_independent():
    works = _population(400, 0.5, 0.5, rho=0.0, seed=7)
    est = estimate_recall(works, ARMS, with_diagnostics=False)
    assert est.estimable
    # Independent arms: the estimate should land near the true population.
    assert 300 <= est.n_hat <= 520


def test_positive_dependence_biases_n_hat_downward():
    """The documented failure mode: dependence makes recall look better than it is."""
    independent = estimate_recall(_population(400, 0.5, 0.5, 0.0, 11), ARMS, with_diagnostics=False)
    dependent = estimate_recall(_population(400, 0.5, 0.5, 0.5, 11), ARMS, with_diagnostics=False)
    assert dependent.n_hat < independent.n_hat


def test_two_arms_declare_independence_untestable():
    """With two arms the assumption cannot be tested -- say so, don't fake a p-value."""
    est = estimate_recall(_population(400, 0.5, 0.5, rho=0.6, seed=3), ARMS)
    joined = " ".join(est.warnings).lower()
    assert "not testable" in joined and "upper bound" in joined
    assert est.detail["assumptions"]["dependence"]["testable"] is False


def test_three_arms_detect_positive_dependence_conditionally():
    """With a third arm, pairwise dependence becomes testable on observed data."""
    rng = random.Random(19)
    works = []
    for i in range(600):
        easy = rng.random() < 0.5
        p = 0.75 if easy else 0.2
        provs = []
        if rng.random() < p:
            provs.append(Provenance("openalex", Direction.BACKWARD, "p", 1, "t", "h"))
        if rng.random() < p:
            provs.append(Provenance("crossref", Direction.FORWARD, "p", 1, "t", "h"))
        if rng.random() < p:
            provs.append(Provenance("semanticscholar", Direction.FORWARD, "p", 1, "t", "h"))
        if not provs:
            continue
        works.append(make_work(f"Study {i}", 2015, f"A{i}", f"10.1000/t{i:04d}").with_provenance(provs))

    arms = [
        {"name": "openalex", "filter": {"provider": "openalex"}},
        {"name": "crossref", "filter": {"provider": "crossref"}},
        {"name": "semanticscholar", "filter": {"provider": "semanticscholar"}},
    ]
    est = estimate_recall(works, arms)
    pairs = est.detail["assumptions"]["dependence"]["pairs"]
    assert pairs and any(p["odds_ratio"] > 1.2 for p in pairs)
    assert any("positively dependent" in w for w in est.warnings)


def test_odds_ratio_is_corrected_for_zero_cells():
    assert odds_ratio(0, 0, 0, 0) is not None


def test_chao1_is_a_lower_bound_and_says_so():
    histories = {f"w{i}": (["a"] if i < 40 else (["b"] if i < 70 else ["a", "b"])) for i in range(100)}
    est = chao1(histories)
    assert est.estimable and est.n_hat >= est.n_observed
    assert any("lower bound" in w for w in est.warnings)


def test_chao1_handles_no_singletons():
    est = chao1({f"w{i}": ["a", "b"] for i in range(10)})
    assert est.estimable and est.recall == 1.0


def test_capture_histories_cover_only_included_works():
    works = [
        make_work("A", 2020, "A", "10.1000/a").with_provenance([_prov(Direction.BACKWARD)]),
        make_work("B", 2020, "B", "10.1000/b").with_provenance([_prov(Direction.FORWARD)]),
        make_work("C", 2020, "C", "10.1000/c").with_provenance(
            [_prov(Direction.BACKWARD), _prov(Direction.FORWARD)]
        ),
    ]
    histories = capture_histories(works, ARMS)
    assert histories["doi:10.1000/c"] == ["backward", "forward"]
    assert len(histories) == 3


def test_provider_arms_are_supported():
    arms = [
        {"name": "openalex", "filter": {"provider": "openalex"}},
        {"name": "crossref", "filter": {"provider": "crossref"}},
    ]
    w = make_work("A", 2020, "A", "10.1000/a").with_provenance(
        [
            Provenance("openalex", Direction.BACKWARD, "p", 1, "t", "h"),
            Provenance("crossref", Direction.BACKWARD, "p", 1, "t", "h"),
        ]
    )
    assert capture_histories([w], arms)["doi:10.1000/a"] == ["crossref", "openalex"]


def test_closure_violation_is_reported():
    works = _population(200, 0.5, 0.5, 0.0, 5)
    est = estimate_recall(works, ARMS, cache_window=("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"))
    assert any("closed" in w for w in est.warnings)
