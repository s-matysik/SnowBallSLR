"""Core immutable data types.

Design invariants enforced here:

* INV-2 -- every :class:`Work` carries complete provenance.
* INV-3 -- every collection exposed to serialization has an explicit total order.
* INV-6 -- provenance is append-only; merging unions provenance, never replaces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "Author",
    "Decision",
    "Direction",
    "Provenance",
    "Work",
    "utcnow",
]


def utcnow() -> datetime:
    """UTC now, second precision.

    Never used for anything that reaches a deterministic artifact: timestamps in
    provenance come from cache entries, not from wall clock at read time.
    """
    return datetime.now(UTC).replace(microsecond=0)


class Direction(StrEnum):
    SEED = "seed"
    BACKWARD = "backward"
    FORWARD = "forward"


class Decision(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNSCREENED = "unscreened"


@dataclass(frozen=True, slots=True, order=True)
class Author:
    surname: str
    given: str | None = None
    orcid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"surname": self.surname, "given": self.given, "orcid": self.orcid}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Author:
        return Author(
            surname=str(d.get("surname") or ""),
            given=d.get("given"),
            orcid=d.get("orcid"),
        )


@dataclass(frozen=True, slots=True, order=True)
class Provenance:
    """Where a record came from. Ordered so provenance tuples sort deterministically."""

    provider: str
    direction: Direction
    parent_key: str | None
    iteration: int
    retrieved_at: str  # ISO-8601 UTC string, taken from the cache entry
    response_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "direction": str(self.direction),
            "parent_key": self.parent_key,
            "iteration": self.iteration,
            "retrieved_at": self.retrieved_at,
            "response_hash": self.response_hash,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Provenance:
        return Provenance(
            provider=str(d["provider"]),
            direction=Direction(d["direction"]),
            parent_key=d.get("parent_key"),
            iteration=int(d["iteration"]),
            retrieved_at=str(d["retrieved_at"]),
            response_hash=str(d["response_hash"]),
        )


@dataclass(frozen=True, slots=True)
class Work:
    """A bibliographic record with full discovery provenance."""

    key: str
    title: str
    title_norm: str
    doi: str | None = None
    abstract: str | None = None
    year: int | None = None
    authors: tuple[Author, ...] = ()
    venue: str | None = None
    type: str | None = None
    language: str | None = None
    is_retracted: bool = False
    unresolved: bool = False
    source_ids: Mapping[str, str] = field(default_factory=dict)
    reference_count: int | None = None
    cited_by_count: int | None = None
    provenance: tuple[Provenance, ...] = ()

    # -- derived helpers -------------------------------------------------

    @property
    def first_author_surname(self) -> str | None:
        return self.authors[0].surname if self.authors else None

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted({p.provider for p in self.provenance}))

    @property
    def directions(self) -> tuple[Direction, ...]:
        return tuple(sorted({p.direction for p in self.provenance}))

    @property
    def parent_keys(self) -> tuple[str, ...]:
        return tuple(sorted({p.parent_key for p in self.provenance if p.parent_key}))

    def with_provenance(self, prov: Iterable[Provenance]) -> Work:
        """Append provenance entries (INV-6: append-only, deduplicated, sorted)."""
        merged = tuple(sorted(set(self.provenance) | set(prov)))
        return replace(self, provenance=merged)

    def merged_with(self, other: Work, precedence: Sequence[str] = ()) -> Work:
        """Merge two records for the same underlying work.

        Field values are taken from the record whose *best* provider ranks
        highest in ``precedence``; ties and missing values fall back to
        "first non-null wins, self first". Conflicts are not silently dropped --
        callers are expected to log them via :func:`field_conflicts`.
        """
        rank_self = _provider_rank(self, precedence)
        rank_other = _provider_rank(other, precedence)
        primary, secondary = (self, other) if rank_self <= rank_other else (other, self)

        def pick(attr: str) -> Any:
            pv = getattr(primary, attr)
            sv = getattr(secondary, attr)
            return pv if pv not in (None, "", ()) else sv

        source_ids: dict[str, str] = dict(secondary.source_ids)
        source_ids.update(dict(primary.source_ids))

        return Work(
            key=min(self.key, other.key),
            title=pick("title"),
            title_norm=pick("title_norm"),
            doi=pick("doi"),
            abstract=pick("abstract"),
            year=pick("year"),
            authors=pick("authors"),
            venue=pick("venue"),
            type=pick("type"),
            language=pick("language"),
            is_retracted=self.is_retracted or other.is_retracted,
            unresolved=self.unresolved and other.unresolved,
            source_ids=dict(sorted(source_ids.items())),
            reference_count=pick("reference_count"),
            cited_by_count=pick("cited_by_count"),
            provenance=tuple(sorted(set(self.provenance) | set(other.provenance))),
        )

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "title_norm": self.title_norm,
            "doi": self.doi,
            "abstract": self.abstract,
            "year": self.year,
            "authors": [a.to_dict() for a in self.authors],
            "venue": self.venue,
            "type": self.type,
            "language": self.language,
            "is_retracted": self.is_retracted,
            "unresolved": self.unresolved,
            "source_ids": dict(sorted(self.source_ids.items())),
            "reference_count": self.reference_count,
            "cited_by_count": self.cited_by_count,
            "provenance": [p.to_dict() for p in sorted(self.provenance)],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Work:
        return Work(
            key=str(d["key"]),
            title=str(d.get("title") or ""),
            title_norm=str(d.get("title_norm") or ""),
            doi=d.get("doi"),
            abstract=d.get("abstract"),
            year=d.get("year"),
            authors=tuple(Author.from_dict(a) for a in d.get("authors") or ()),
            venue=d.get("venue"),
            type=d.get("type"),
            language=d.get("language"),
            is_retracted=bool(d.get("is_retracted", False)),
            unresolved=bool(d.get("unresolved", False)),
            source_ids=dict(sorted((d.get("source_ids") or {}).items())),
            reference_count=d.get("reference_count"),
            cited_by_count=d.get("cited_by_count"),
            provenance=tuple(
                sorted(Provenance.from_dict(p) for p in d.get("provenance") or ())
            ),
        )


def _provider_rank(work: Work, precedence: Sequence[str]) -> int:
    """Lowest (best) precedence index among the providers that saw this record."""
    if not precedence:
        return 0
    ranks = [
        precedence.index(p) if p in precedence else len(precedence)
        for p in work.providers
    ]
    return min(ranks) if ranks else len(precedence)


def field_conflicts(a: Work, b: Work) -> dict[str, tuple[Any, Any]]:
    """Fields where two records for the same work disagree (both non-null)."""
    out: dict[str, tuple[Any, Any]] = {}
    for attr in ("title_norm", "doi", "year", "venue", "type", "language"):
        av, bv = getattr(a, attr), getattr(b, attr)
        if av not in (None, "") and bv not in (None, "") and av != bv:
            out[attr] = (av, bv)
    return out
