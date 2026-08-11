"""Semantic Scholar provider -- both directions.

Without an API key the public endpoint is limited to roughly one request per
second; the limiter defaults accordingly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..identity.keys import canonical_key
from ..identity.normalize import normalize_doi, normalize_title, normalize_year
from ..types import Author, Direction, Work
from .base import BaseProvider, dedupe_preserving_order

__all__ = ["SemanticScholarProvider", "parse_work"]

_FIELDS = (
    "paperId,corpusId,externalIds,title,abstract,year,venue,publicationTypes,"
    "referenceCount,citationCount"
)


def parse_work(raw: Mapping[str, Any] | None) -> Work | None:
    if not raw or not isinstance(raw, Mapping):
        return None
    ext = raw.get("externalIds") or {}
    doi = normalize_doi(ext.get("DOI"))
    title = str(raw.get("title") or "").strip()
    if not title and not doi:
        return None
    year = normalize_year(raw.get("year"))

    authors: list[Author] = []
    for a in raw.get("authors") or []:
        display = str(a.get("name") or "").strip()
        if not display:
            continue
        parts = display.split()
        authors.append(
            Author(surname=parts[-1], given=" ".join(parts[:-1]) or None)
        )

    source_ids: dict[str, str] = {}
    corpus = raw.get("corpusId")
    if corpus:
        source_ids["s2"] = str(corpus)
    if ext.get("PubMed"):
        source_ids["pmid"] = str(ext["PubMed"])

    types = raw.get("publicationTypes") or []
    key = canonical_key(
        doi=doi,
        source_ids=source_ids,
        title=title,
        year=year,
        first_author_surname=authors[0].surname if authors else None,
    )
    return Work(
        key=key,
        title=title,
        title_norm=normalize_title(title),
        doi=doi,
        abstract=raw.get("abstract"),
        year=year,
        authors=tuple(authors),
        venue=raw.get("venue") or None,
        type=str(types[0]).lower() if types else None,
        source_ids=source_ids,
        reference_count=raw.get("referenceCount"),
        cited_by_count=raw.get("citationCount"),
    )


class SemanticScholarProvider(BaseProvider):
    name = "semanticscholar"
    supports = frozenset({Direction.BACKWARD, Direction.FORWARD})

    def __init__(
        self,
        cache,
        *,
        api_key_env: str = "S2_API_KEY",
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(cache, **kwargs)
        self.api_key = os.environ.get(api_key_env) or None
        self.base_url = base_url.rstrip("/")

    def headers(self) -> dict[str, str]:
        h = super().headers()
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _paper_id(self, work: Work) -> str | None:
        if work.doi:
            return f"DOI:{work.doi}"
        s2 = work.source_ids.get("s2")
        return f"CorpusId:{s2}" if s2 else None

    def resolve(self, ident: str) -> Work | None:
        pid = ident if ":" in ident else f"DOI:{ident}"
        body = self.fetch(
            "resolve",
            {"fields": _FIELDS + ",authors", "_id": pid},
            f"{self.base_url}/paper/{pid}",
        )
        return parse_work(body)

    def _paged(self, work: Work, endpoint: str, container: str) -> list[Work]:
        pid = self._paper_id(work)
        if not pid:
            return []
        out: list[Work] = []
        offset = 0
        limit = 100
        while True:
            params = {
                "fields": ",".join(f"{container}.{f}" for f in _FIELDS.split(","))
                + f",{container}.authors",
                "offset": offset,
                "limit": limit,
            }
            body = self.fetch(
                endpoint,
                {**params, "_id": pid},
                f"{self.base_url}/paper/{pid}/{endpoint}",
            )
            if not body:
                break
            data = body.get("data") or []
            for row in data:
                w = parse_work(row.get(container))
                if w:
                    out.append(w)
            nxt = body.get("next")
            if not nxt or not data:
                break
            offset = int(nxt)
            if offset > 10000:  # hard safety bound
                break
        return dedupe_preserving_order(out)

    def references(self, work: Work) -> list[Work]:
        return self._paged(work, "references", "citedPaper")

    def citations(self, work: Work) -> list[Work]:
        return self._paged(work, "citations", "citingPaper")
