"""Offline provider: replays a frozen citation graph.

Backs the test suite (INV-4: no network in CI) and the frozen-snapshot
simulation harness. The fixture format is a single JSON document::

    {
      "works":      {"doi:10.1/a": {...work fields...}},
      "references": {"doi:10.1/a": ["doi:10.1/b", ...]},
      "citations":  {"doi:10.1/a": ["doi:10.1/c", ...]}
    }
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..types import Direction, Work
from .base import BaseProvider, dedupe_preserving_order

__all__ = ["OfflineProvider", "load_graph"]


def load_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class OfflineProvider(BaseProvider):
    name = "offline"
    supports = frozenset({Direction.BACKWARD, Direction.FORWARD})

    def __init__(self, cache, *, graph: Mapping[str, Any] | str | Path, **kwargs: Any) -> None:
        super().__init__(cache, **kwargs)
        data = load_graph(graph) if isinstance(graph, (str, Path)) else dict(graph)
        self._works: dict[str, Work] = {
            k: Work.from_dict({**v, "key": k}) for k, v in (data.get("works") or {}).items()
        }
        self._references: dict[str, list[str]] = {
            k: list(v) for k, v in (data.get("references") or {}).items()
        }
        self._citations: dict[str, list[str]] = {
            k: list(v) for k, v in (data.get("citations") or {}).items()
        }
        # Aliases let fixtures reference works by DOI or by raw identifier.
        self._alias: dict[str, str] = {}
        for key, w in self._works.items():
            self._alias[key] = key
            if w.doi:
                self._alias[w.doi] = key
                self._alias[f"doi:{w.doi}"] = key
            for ns, val in w.source_ids.items():
                self._alias[f"{ns}:{val}"] = key
                self._alias[val] = key

    def _lookup(self, ident: str) -> Work | None:
        key = self._alias.get(ident) or self._alias.get(ident.lower())
        return self._works.get(key) if key else None

    def resolve(self, ident: str) -> Work | None:
        return self._lookup(ident)

    def _neighbours(self, work: Work, table: dict[str, list[str]]) -> list[Work]:
        keys = table.get(work.key) or []
        out = [self._works[k] for k in keys if k in self._works]
        return dedupe_preserving_order(out)

    def references(self, work: Work) -> list[Work]:
        return self._neighbours(work, self._references)

    def citations(self, work: Work) -> list[Work]:
        return self._neighbours(work, self._citations)

    @property
    def last_response_hash(self) -> str:
        return "sha256:offline"

    @property
    def last_retrieved_at(self) -> str:
        return "1970-01-01T00:00:00Z"
