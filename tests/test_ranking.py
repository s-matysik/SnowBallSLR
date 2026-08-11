"""Ranking produces a total order and never touches inclusion (INV-5)."""

from __future__ import annotations

import random

from snowballslr.config import RankingSettings
from snowballslr.ranking import (
    BM25Ranker,
    NetworkRanker,
    NullRanker,
    RRFFusion,
    build_ranker,
)
from snowballslr.ranking.fusion import rank_positions
from snowballslr.types import Direction, Provenance
from tests.conftest import make_work


def _with_parent(work, parent, direction=Direction.BACKWARD):
    return work.with_provenance(
        [
            Provenance(
                provider="offline",
                direction=direction,
                parent_key=parent,
                iteration=1,
                retrieved_at="2026-01-01T00:00:00Z",
                response_hash="sha256:x",
            )
        ]
    )


def test_bm25_prefers_topically_similar_records():
    included = [make_work("Snowballing for systematic literature reviews", 2020, "Wohlin")]
    candidates = [
        make_work("Citation searching in systematic reviews", 2021, "A"),
        make_work("Catalysis of palladium complexes", 2019, "B"),
    ]
    ranker = BM25Ranker()
    ranker.fit(included)
    scores = ranker.score(candidates)
    assert scores[candidates[0].key] > scores[candidates[1].key]


def test_network_ranker_prefers_multiply_connected_candidates():
    parents = [make_work(f"Parent {i}", 2020, "P", f"10.1000/p{i}") for i in range(3)]
    strong = _with_parent(_with_parent(make_work("Strong", 2021, "S"), parents[0].key), parents[1].key)
    weak = _with_parent(make_work("Weak", 2021, "W"), parents[2].key)
    ranker = NetworkRanker()
    ranker.fit(parents)
    scores = ranker.score([strong, weak])
    assert scores[strong.key] > scores[weak.key]


def test_rank_positions_break_ties_by_key():
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    assert rank_positions(scores) == {"a": 1, "b": 2, "c": 3}


def test_ranking_is_order_independent():
    included = [make_work("Snowballing systematic reviews", 2020, "W")]
    candidates = [make_work(f"Study number {i} on citation searching", 2020, "A") for i in range(8)]
    ranker = build_ranker(RankingSettings())
    ranker.fit(included)
    baseline = rank_positions(dict(ranker.score(candidates)))
    for trial in range(5):
        shuffled = candidates[:]
        random.Random(trial).shuffle(shuffled)
        assert rank_positions(dict(ranker.score(shuffled))) == baseline


def test_rrf_combines_components_deterministically():
    included = [make_work("Snowballing systematic reviews", 2020, "W", "10.1000/w")]
    candidates = [make_work(f"Citation searching study {i}", 2020, "A") for i in range(5)]
    fusion = RRFFusion([BM25Ranker(), NetworkRanker()])
    fusion.fit(included)
    assert dict(fusion.score(candidates)) == dict(fusion.score(candidates))


def test_null_ranker_scores_everything_zero():
    candidates = [make_work("X", 2020, "A"), make_work("Y", 2020, "B")]
    assert set(NullRanker().score(candidates).values()) == {0.0}
