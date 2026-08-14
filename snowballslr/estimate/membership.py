"""Membership arms: capture arms defined by presence in an external record set.

The estimators in this package need at least two arms that can each capture the
*same* record. Provider arms satisfy that; direction arms do not, because a
citation graph is temporally acyclic (see ``Config._arm_design_warnings``). Both
provider arms are nonetheless produced by the same crawl, so a third arm drawn
from a different discovery process is what makes arm independence testable
rather than assumed: with two arms a 2x2 table has no residual degree of
freedom, so no model selection is possible.

A *membership arm* is that third arm. It is defined by a set of identifiers
exported from a database the crawl did not use -- typically a Scopus or Web of
Science query result. A record is "captured" by the arm when its identifier
appears in that set.

The set is a snapshot, and the arm inherits whatever the query that produced it
captured. That is a real limitation of the design: a membership arm is not
independent of how the topic was defined, and an estimate resting on one should
say so. It is still a different indexing pipeline with different coverage
decisions from the crawl's own providers, which is the property the estimator
needs.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MembershipSet", "load_membership_sets", "normalize_doi"]


def normalize_doi(raw: str | None) -> str | None:
    """Fold a DOI to its comparable form.

    Accepts bare DOIs, ``doi:`` keys and resolver URLs. DOIs are
    case-insensitive by specification, so comparison is done in lower case.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.strip().rstrip(".")
    return s if s.startswith("10.") else None


@dataclass(frozen=True)
class MembershipSet:
    """A named set of DOIs standing in for one capture arm.

    ``n_rows`` and ``n_without_doi`` are retained so that a report can state
    what fraction of the source export the arm actually represents: an export
    with poor identifier coverage yields an arm that under-captures for a
    mechanical reason rather than a bibliographic one.
    """

    name: str
    dois: frozenset[str]
    source: str
    n_rows: int = 0
    n_without_doi: int = 0

    def __contains__(self, doi: str | None) -> bool:
        norm = normalize_doi(doi)
        return norm is not None and norm in self.dois

    @property
    def coverage(self) -> float:
        """Share of source rows that carried a usable DOI."""
        return (self.n_rows - self.n_without_doi) / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "n_dois": len(self.dois),
            "n_rows": self.n_rows,
            "n_without_doi": self.n_without_doi,
            "coverage": round(self.coverage, 4),
        }

    @staticmethod
    def from_csv(
        name: str, path: str | Path, *, doi_column: str = "DOI"
    ) -> MembershipSet:
        """Build an arm from a database export.

        Reads ``doi_column`` from a CSV export (the column Scopus and Web of
        Science both emit). Rows without a usable DOI are counted, not dropped
        silently: that count is the arm's mechanical coverage ceiling.
        """
        path = Path(path)
        # Reference lists in bibliographic exports routinely exceed the default
        # field limit; without this a long-reference row raises mid-parse.
        limit = csv.field_size_limit()
        try:
            csv.field_size_limit(sys.maxsize)
            with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or doi_column not in reader.fieldnames:
                    raise ValueError(
                        f"{path.name} has no '{doi_column}' column; "
                        f"found {reader.fieldnames}"
                    )
                dois: set[str] = set()
                n_rows = n_missing = 0
                for row in reader:
                    n_rows += 1
                    norm = normalize_doi(row.get(doi_column))
                    if norm is None:
                        n_missing += 1
                    else:
                        dois.add(norm)
        finally:
            csv.field_size_limit(limit)
        return MembershipSet(
            name=name,
            dois=frozenset(dois),
            source=str(path),
            n_rows=n_rows,
            n_without_doi=n_missing,
        )

    @staticmethod
    def from_dois(name: str, dois: Iterable[str], *, source: str = "") -> MembershipSet:
        norm = {d for d in (normalize_doi(x) for x in dois) if d}
        return MembershipSet(
            name=name, dois=frozenset(norm), source=source, n_rows=len(norm)
        )


def load_membership_sets(
    spec: Mapping[str, str], *, doi_column: str = "DOI"
) -> dict[str, MembershipSet]:
    """Load every configured membership set, keyed by arm-filter value."""
    return {
        name: MembershipSet.from_csv(name, path, doi_column=doi_column)
        for name, path in spec.items()
    }
