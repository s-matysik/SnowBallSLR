"""Deduplication invariants: idempotent, order-independent, guarded."""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from snowballslr.identity.dedup import deduplicate
from tests.conftest import make_work


def test_doi_variants_merge_to_one_record():
    works = [
        make_work("Deep Learning for X", 2020, "Kowalski", "10.1000/abc"),
        make_work("Deep learning for X.", 2021, "Kowalski", "https://doi.org/10.1000/ABC"),
        make_work("Something Else", 2019, "Nowak"),
    ]
    result = deduplicate(works)
    assert len(result.works) == 2
    assert result.works[0].key == "doi:10.1000/abc"


def test_fuzzy_title_merges_under_t3():
    works = [
        make_work("Snowballing in systematic reviews", 2020, "Wohlin"),
        make_work("Snowballing in systematic review", 2021, "Wohlin"),
    ]
    result = deduplicate(works, jw_threshold=0.95)
    assert len(result.works) == 1
    assert result.merges[0].tier in ("T2", "T3")


def test_distinct_dois_never_merge():
    works = [
        make_work("Same Title", 2020, "Smith", "10.1000/a"),
        make_work("Same Title", 2020, "Smith", "10.1000/b"),
    ]
    assert len(deduplicate(works).works) == 2


def test_oversized_cluster_is_rejected_not_merged():
    works = [make_work("Editorial", 2020, "Smith") for _ in range(3)]
    works += [
        make_work("Editorial", 2020, "Smith", f"10.1000/x{i}") for i in range(12)
    ]
    result = deduplicate(works, max_cluster_size=4)
    assert result.suspicious_clusters
    assert len(result.works) > 1


def test_dedup_is_idempotent():
    works = [
        make_work("Alpha study", 2020, "A", "10.1000/a"),
        make_work("Alpha study.", 2020, "A"),
        make_work("Beta study", 2019, "B"),
    ]
    once = deduplicate(works)
    twice = deduplicate(list(once.works))
    assert [w.key for w in twice.works] == [w.key for w in once.works]


@pytest.mark.parametrize("trial", range(25))
def test_dedup_is_order_independent(trial):
    works = [
        make_work("Alpha study", 2020, "A", "10.1000/a"),
        make_work("Alpha study.", 2020, "A"),
        make_work("Beta study", 2019, "B"),
        make_work("Gamma work", 2021, "C", "10.1000/c"),
        make_work("Gamma work", 2021, "C"),
    ]
    baseline = [w.key for w in deduplicate(works).works]
    shuffled = works[:]
    random.Random(trial).shuffle(shuffled)
    assert [w.key for w in deduplicate(shuffled).works] == baseline


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=30),
            st.integers(min_value=1990, max_value=2026),
            st.text(alphabet="abcdefgh", min_size=1, max_size=6),
        ),
        min_size=1,
        max_size=12,
    ),
    st.randoms(),
)
def test_dedup_order_independence_property(rows, rng):
    works = [make_work(t, y, s) for t, y, s in rows]
    baseline = [w.key for w in deduplicate(works).works]
    shuffled = works[:]
    rng.shuffle(shuffled)
    assert [w.key for w in deduplicate(shuffled).works] == baseline


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20),
            st.integers(min_value=1990, max_value=2026),
            st.text(alphabet="abcd", min_size=1, max_size=4),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_dedup_never_grows_the_set(rows):
    works = [make_work(t, y, s) for t, y, s in rows]
    assert len(deduplicate(works).works) <= len(works)
