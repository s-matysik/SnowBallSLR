"""Crossref provider -- backward direction only.

Crossref exposes ``reference`` arrays but no citing-works endpoint. Reference
arrays are frequently incomplete for older and non-STEM literature, which is
precisely the coverage gap the GROBID provider exists to fill.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..identity.keys import canonical_key
from ..identity.normalize import normalize_doi, normalize_title, normalize_year
from ..types import Author, Direction, Work
from .base import BaseProvider, dedupe_preserving_order

__all__ = ["CrossrefProvider", "parse_item", "parse_reference"]


def _authors(raw: Mapping[str, Any]) -> tuple[Author, ...]:
    out: list[Author] = []
    for a in raw.get("author") or []:
        surname = str(a.get("family") or "").strip()
        if not surname:
            continue
        out.append(
            Author(surname=surname, given=a.get("given"), orcid=a.get("ORCID"))
        )
    return tuple(out)


def _year(raw: Mapping[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued", "created"):
        block = raw.get(field) or {}
        parts = block.get("date-parts") or []
        if parts and parts[0]:
            y = normalize_year(parts[0][0])
            if y:
                return y
    return None


def parse_item(raw: Mapping[str, Any] | None) -> Work | None:
    """Parse a full Crossref work item."""
    if not raw or not isinstance(raw, Mapping):
        return None
    titles = raw.get("title") or []
    title = str(titles[0]).strip() if titles else ""
    doi = normalize_doi(raw.get("DOI"))
    if not title and not doi:
        return None
    authors = _authors(raw)
    containers = raw.get("container-title") or []
    key = canonical_key(
        doi=doi,
        title=title,
        year=_year(raw),
        first_author_surname=authors[0].surname if authors else None,
    )
    return Work(
        key=key,
        title=title,
        title_norm=normalize_title(title),
        doi=doi,
        abstract=raw.get("abstract"),
        year=_year(raw),
        authors=authors,
        venue=str(containers[0]) if containers else None,
        type=raw.get("type"),
        language=raw.get("language"),
        source_ids={},
        reference_count=raw.get("reference-count"),
        cited_by_count=raw.get("is-referenced-by-count"),
    )


def parse_reference(raw: Mapping[str, Any]) -> Work | None:
    """Parse one entry of a Crossref ``reference`` array.

    Reference entries are sparse: often a DOI and little else, or an unstructured
    string. Records without a resolvable DOI are returned flagged ``unresolved``.
    """
    doi = normalize_doi(raw.get("DOI"))
    title = str(
        raw.get("article-title") or raw.get("volume-title") or raw.get("unstructured") or ""
    ).strip()
    year = normalize_year(raw.get("year"))
    surname = str(raw.get("author") or "").strip() or None
    if not doi and not title:
        return None
    key = canonical_key(doi=doi, title=title, year=year, first_author_surname=surname)
    return Work(
        key=key,
        title=title,
        title_norm=normalize_title(title),
        doi=doi,
        year=year,
        authors=(Author(surname=surname),) if surname else (),
        venue=raw.get("journal-title"),
        unresolved=doi is None,
    )


class CrossrefProvider(BaseProvider):
    name = "crossref"
    supports = frozenset({Direction.BACKWARD})

    def __init__(
        self,
        cache,
        *,
        mailto: str | None = None,
        base_url: str = "https://api.crossref.org",
        **kwargs: Any,
    ) -> None:
        super().__init__(cache, **kwargs)
        self.mailto = mailto
        self.base_url = base_url.rstrip("/")

    def headers(self) -> dict[str, str]:
        h = super().headers()
        if self.mailto:
            h["User-Agent"] = f"SnowBallSLR/1.0 (mailto:{self.mailto})"
        return h

    def _params(self) -> dict[str, Any]:
        return {"mailto": self.mailto} if self.mailto else {}

    def resolve(self, ident: str) -> Work | None:
        doi = normalize_doi(ident)
        if not doi:
            return None
        body = self.fetch(
            "resolve", {**self._params(), "_doi": doi}, f"{self.base_url}/works/{doi}"
        )
        if not body:
            return None
        return parse_item(body.get("message"))

    def references(self, work: Work) -> list[Work]:
        if not work.doi:
            return []
        body = self.fetch(
            "resolve",
            {**self._params(), "_doi": work.doi},
            f"{self.base_url}/works/{work.doi}",
        )
        if not body:
            return []
        message = body.get("message") or {}
        out: list[Work] = []
        for ref in message.get("reference") or []:
            w = parse_reference(ref)
            if w is not None:
                out.append(w)
        return dedupe_preserving_order(out)

    def citations(self, work: Work) -> list[Work]:
        return []

    def match_bibliographic(
        self, query: str, *, rows: int = 3
    ) -> list[tuple[Work, float]]:
        """Resolve a free-text reference string to candidate works with scores.

        Used by the GROBID provider to attach DOIs to extracted references.
        """
        params = {
            **self._params(),
            "query.bibliographic": query,
            "rows": rows,
            "select": "DOI,title,author,issued,container-title,type",
        }
        body = self.fetch("match", params, f"{self.base_url}/works")
        if not body:
            return []
        items = (body.get("message") or {}).get("items") or []
        out: list[tuple[Work, float]] = []
        for item in items:
            w = parse_item(item)
            if w is not None:
                out.append((w, float(item.get("score", 0.0))))
        return out
