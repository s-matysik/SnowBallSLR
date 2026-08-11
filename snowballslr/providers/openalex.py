"""OpenAlex provider -- primary source, both directions.

Backward = ``referenced_works``; forward = ``filter=cites:<id>`` with cursor
pagination. Abstracts arrive as an inverted index and are reconstructed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ProviderError
from ..identity.keys import canonical_key, normalize_source_id
from ..identity.normalize import normalize_doi, normalize_title, normalize_year
from ..types import Author, Direction, Work
from .base import BaseProvider, dedupe_preserving_order

__all__ = ["OpenAlexProvider", "invert_abstract", "parse_work"]

_SELECT = (
    "id,doi,title,display_name,publication_year,type,language,authorships,"
    "primary_location,referenced_works,cited_by_count,referenced_works_count,"
    "abstract_inverted_index,is_retracted,updated_date"
)


def invert_abstract(index: Mapping[str, list[int]] | None) -> str | None:
    """Reconstruct plain text from OpenAlex's inverted abstract index."""
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for token, locs in index.items():
        for loc in locs:
            positions.append((int(loc), token))
    if not positions:
        return None
    positions.sort()
    return " ".join(tok for _, tok in positions)


def parse_work(raw: Mapping[str, Any] | None) -> Work | None:
    if not raw or not isinstance(raw, Mapping):
        return None
    oa_id = normalize_source_id("openalex", raw.get("id"))
    doi = normalize_doi(raw.get("doi"))
    title = str(raw.get("title") or raw.get("display_name") or "").strip()
    year = normalize_year(raw.get("publication_year"))

    authors: list[Author] = []
    for a in raw.get("authorships") or []:
        person = a.get("author") or {}
        display = str(person.get("display_name") or "").strip()
        if not display:
            continue
        parts = display.split()
        surname = parts[-1] if parts else display
        given = " ".join(parts[:-1]) or None
        authors.append(
            Author(surname=surname, given=given, orcid=person.get("orcid"))
        )

    loc = raw.get("primary_location") or {}
    source = (loc.get("source") or {}) if isinstance(loc, Mapping) else {}
    venue = source.get("display_name")

    source_ids: dict[str, str] = {}
    if oa_id:
        source_ids["openalex"] = oa_id

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
        abstract=invert_abstract(raw.get("abstract_inverted_index")),
        year=year,
        authors=tuple(authors),
        venue=venue,
        type=raw.get("type"),
        language=raw.get("language"),
        is_retracted=bool(raw.get("is_retracted", False)),
        source_ids=source_ids,
        reference_count=raw.get("referenced_works_count"),
        cited_by_count=raw.get("cited_by_count"),
    )


class OpenAlexProvider(BaseProvider):
    name = "openalex"
    supports = frozenset({Direction.BACKWARD, Direction.FORWARD})

    def __init__(
        self,
        cache,
        *,
        mailto: str | None = None,
        base_url: str = "https://api.openalex.org",
        per_page: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(cache, **kwargs)
        self.mailto = mailto
        self.base_url = base_url.rstrip("/")
        self.per_page = per_page
        self._id_filter_key: str | None = None

    def headers(self) -> dict[str, str]:
        h = super().headers()
        if self.mailto:
            h["User-Agent"] = f"SnowBallSLR/1.0 (mailto:{self.mailto})"
        return h

    def version_from_response(self, resp) -> dict[str, Any]:
        return {"date": resp.headers.get("date", "")}

    def _base_params(self) -> dict[str, Any]:
        p: dict[str, Any] = {"select": _SELECT}
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    # -- interface -------------------------------------------------------

    def resolve(self, ident: str) -> Work | None:
        oa = normalize_source_id("openalex", ident)
        if oa:
            url = f"{self.base_url}/works/{oa}"
            params = self._base_params()
        else:
            url = f"{self.base_url}/works/doi:{ident}"
            params = self._base_params()
        body = self.fetch("resolve", {**params, "_id": ident}, url)
        work = parse_work(body)
        return work

    def references(self, work: Work) -> list[Work]:
        oa = work.source_ids.get("openalex")
        if not oa:
            return []
        url = f"{self.base_url}/works/{oa}"
        body = self.fetch("resolve", {**self._base_params(), "_id": oa}, url)
        if not body:
            return []
        ref_ids = [
            normalize_source_id("openalex", r) for r in body.get("referenced_works") or []
        ]
        ref_ids = [r for r in ref_ids if r]
        if not ref_ids:
            return []
        return self._fetch_by_ids(sorted(ref_ids))

    def citations(self, work: Work) -> list[Work]:
        oa = work.source_ids.get("openalex")
        if not oa:
            return []
        out: list[Work] = []
        cursor = "*"
        page = 0
        while cursor:
            params = {
                **self._base_params(),
                "filter": f"cites:{oa}",
                "per-page": self.per_page,
                "cursor": cursor,
            }
            body = self.fetch(
                "citations", {**params, "_page": page}, f"{self.base_url}/works"
            )
            if not body:
                break
            for raw in body.get("results") or []:
                w = parse_work(raw)
                if w:
                    out.append(w)
            cursor = ((body.get("meta") or {}).get("next_cursor")) or None
            page += 1
            if page > 200:  # hard safety bound
                break
        return dedupe_preserving_order(out)

    # OpenAlex accepts up to 50 pipe-separated values per filter. The documented
    # attribute filter is ``ids.openalex``; official tutorials also use the bare
    # ``openalex`` alias. Both are tried, because a wrong key returns HTTP 400
    # rather than an empty result, and the accepted alias has changed before.
    _ID_FILTER_KEYS = ("ids.openalex", "openalex")

    def _fetch_by_ids(self, ids: list[str]) -> list[Work]:
        out: list[Work] = []
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            joined = "|".join(batch)
            body = None
            errors: list[str] = []
            keys = (
                (self._id_filter_key,) if self._id_filter_key else self._ID_FILTER_KEYS
            )
            for key in keys:
                params = {
                    **self._base_params(),
                    "filter": f"{key}:{joined}",
                    "per-page": 50,
                }
                try:
                    body = self.fetch(
                        "batch",
                        {**params, "_batch": joined},
                        f"{self.base_url}/works",
                    )
                except ProviderError as exc:
                    errors.append(f"{key} -> {exc}")
                    continue
                self._id_filter_key = key  # remember the alias that works
                break
            if body is None:
                raise ProviderError(
                    "openalex: no supported identifier filter. Tried "
                    + " || ".join(errors)
                )
            for raw in body.get("results") or []:
                w = parse_work(raw)
                if w:
                    out.append(w)
        return dedupe_preserving_order(out)
