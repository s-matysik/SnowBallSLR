"""Provenance is append-only and merging is order-insensitive."""

from __future__ import annotations

from snowballslr.types import Direction, Provenance, Work
from tests.conftest import make_work


def _prov(provider: str, direction: Direction, parent: str | None = None) -> Provenance:
    return Provenance(
        provider=provider,
        direction=direction,
        parent_key=parent,
        iteration=1,
        retrieved_at="2026-01-01T00:00:00Z",
        response_hash="sha256:x",
    )


def test_provenance_accumulates_and_never_replaces():
    w = make_work("A study", 2020, "Smith", "10.1000/a")
    w1 = w.with_provenance([_prov("openalex", Direction.BACKWARD, "p1")])
    w2 = w1.with_provenance([_prov("crossref", Direction.FORWARD, "p2")])
    assert len(w2.provenance) == 2
    assert w2.providers == ("crossref", "openalex")
    assert set(w2.directions) == {Direction.BACKWARD, Direction.FORWARD}
    assert w2.parent_keys == ("p1", "p2")


def test_duplicate_provenance_is_deduplicated():
    w = make_work("A study", 2020, "Smith", "10.1000/a")
    p = _prov("openalex", Direction.BACKWARD, "p1")
    assert len(w.with_provenance([p, p]).provenance) == 1


def test_merge_is_commutative_on_provenance_and_ids():
    a = make_work("A study", 2020, "Smith", "10.1000/a").with_provenance(
        [_prov("openalex", Direction.BACKWARD, "p1")]
    )
    b = make_work("A study", 2020, "Smith", "10.1000/a").with_provenance(
        [_prov("crossref", Direction.FORWARD, "p2")]
    )
    ab, ba = a.merged_with(b), b.merged_with(a)
    assert ab.provenance == ba.provenance
    assert ab.key == ba.key


def test_precedence_decides_field_conflicts():
    a = Work(key="k", title="From OpenAlex", title_norm="from openalex", venue="OA venue")
    a = a.with_provenance([_prov("openalex", Direction.BACKWARD)])
    b = Work(key="k", title="From Crossref", title_norm="from crossref", venue="CR venue")
    b = b.with_provenance([_prov("crossref", Direction.BACKWARD)])
    merged = b.merged_with(a, precedence=["openalex", "crossref"])
    assert merged.venue == "OA venue"


def test_roundtrip_serialization():
    w = make_work("A study", 2020, "Smith", "10.1000/a").with_provenance(
        [_prov("openalex", Direction.BACKWARD, "p1")]
    )
    assert Work.from_dict(w.to_dict()) == w
