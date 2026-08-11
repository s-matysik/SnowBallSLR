"""Gold-standard oracle for simulation mode.

Simulation is how the stopping rules get validated: with a complete label set we
can measure, for every rule, the recall actually achieved at the moment it fired
and how far that sits from the oracle-optimal stop.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..identity.keys import canonical_key
from ..identity.normalize import normalize_doi
from ..types import Decision

__all__ = ["Oracle", "load_gold_standard"]


def load_gold_standard(path: str | Path) -> set[str]:
    """Read a gold-standard inclusion list (CSV with ``key`` or ``doi``, or JSON)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows: Iterable[Mapping[str, str]] = (
            data if isinstance(data, list) else data.get("included", [])
        )
        return {_key_of(r) for r in rows if _key_of(r)}
    out: set[str] = set()
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            k = _key_of(row)
            if k:
                out.add(k)
    return out


def _key_of(row: Mapping[str, str]) -> str | None:
    key = (row.get("key") or "").strip()
    if key:
        return key
    doi = normalize_doi(row.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = (row.get("title") or "").strip()
    if title:
        return canonical_key(
            title=title, year=row.get("year"), first_author_surname=row.get("author")
        )
    return None


@dataclass
class Oracle:
    """Answers screening decisions from a known inclusion set."""

    included: frozenset[str]

    @staticmethod
    def from_file(path: str | Path) -> Oracle:
        return Oracle(frozenset(load_gold_standard(path)))

    @staticmethod
    def from_keys(keys: Iterable[str]) -> Oracle:
        return Oracle(frozenset(keys))

    def decide(self, key: str) -> str:
        return Decision.INCLUDE.value if key in self.included else Decision.EXCLUDE.value

    def as_mapping(self, keys: Sequence[str] | None = None) -> dict[str, str]:
        if keys is None:
            return {k: Decision.INCLUDE.value for k in sorted(self.included)}
        return {k: self.decide(k) for k in sorted(keys)}

    def __len__(self) -> int:
        return len(self.included)

    def __contains__(self, key: object) -> bool:
        return key in self.included
