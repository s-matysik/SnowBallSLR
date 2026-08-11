"""GROBID provider -- backward direction from local PDFs.

This is the coverage differentiator: OpenAlex reference coverage is roughly 83%
on recent DOI-bearing literature and materially lower for older, non-English and
non-STEM work (Culbert et al. 2025). Extracting bibliographies from the PDFs the
reviewer already holds fills exactly that gap.

Extracted references are matched to DOIs through Crossref
``query.bibliographic``; matches are accepted only when the Crossref score, the
normalized-title similarity and the year all clear their thresholds. Unmatched
references are retained but flagged ``unresolved`` and never expanded forward.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rapidfuzz.distance import JaroWinkler

from ..identity.keys import canonical_key
from ..identity.normalize import normalize_title, normalize_year
from ..types import Author, Direction, Work
from .base import BaseProvider, dedupe_preserving_order

__all__ = ["GrobidProvider", "load_pdf_map", "parse_tei_references"]

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def load_pdf_map(path: str | Path) -> dict[str, str]:
    """Read ``pdfs.csv`` mapping canonical keys to local PDF paths."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("key") or "").strip()
            pdf = (row.get("path") or "").strip()
            if key and pdf:
                out[key] = pdf
    return dict(sorted(out.items()))


def parse_tei_references(tei_xml: str) -> list[dict[str, Any]]:
    """Extract structured references from a GROBID TEI document."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []
    out: list[dict[str, Any]] = []
    for bibl in root.iterfind(".//tei:listBibl/tei:biblStruct", _TEI_NS):
        title_el = bibl.find(".//tei:title[@level='a']", _TEI_NS)
        if title_el is None:
            title_el = bibl.find(".//tei:title", _TEI_NS)
        title = (title_el.text or "").strip() if title_el is not None else ""

        authors: list[str] = []
        for pers in bibl.iterfind(".//tei:author/tei:persName", _TEI_NS):
            surname_el = pers.find("tei:surname", _TEI_NS)
            if surname_el is not None and surname_el.text:
                authors.append(surname_el.text.strip())

        year = None
        date_el = bibl.find(".//tei:date[@type='published']", _TEI_NS)
        if date_el is not None:
            year = normalize_year(date_el.get("when") or date_el.text)

        journal_el = bibl.find(".//tei:title[@level='j']", _TEI_NS)
        doi_el = bibl.find(".//tei:idno[@type='DOI']", _TEI_NS)

        if not title and not authors:
            continue
        out.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "venue": (journal_el.text or "").strip() if journal_el is not None else None,
                "doi": (doi_el.text or "").strip() if doi_el is not None else None,
            }
        )
    return out


class GrobidProvider(BaseProvider):
    name = "grobid"
    supports = frozenset({Direction.BACKWARD})

    def __init__(
        self,
        cache,
        *,
        url: str = "http://localhost:8070",
        pdf_map: Mapping[str, str] | str | Path | None = None,
        crossref: Any = None,
        min_match_score: float = 60.0,
        min_title_similarity: float = 0.90,
        **kwargs: Any,
    ) -> None:
        super().__init__(cache, **kwargs)
        self.url = url.rstrip("/")
        if isinstance(pdf_map, (str, Path)):
            self.pdf_map = load_pdf_map(pdf_map)
        else:
            self.pdf_map = dict(sorted((pdf_map or {}).items()))
        self.crossref = crossref
        self.min_match_score = min_match_score
        self.min_title_similarity = min_title_similarity
        self.match_stats = {"attempted": 0, "matched": 0, "unresolved": 0}

    # -- extraction ------------------------------------------------------

    def _process_pdf(self, key: str, pdf_path: str) -> list[dict[str, Any]]:
        cache_key_params = {"_key": key, "_pdf": Path(pdf_path).name}

        def fetch() -> tuple[Any, int, dict[str, Any]]:
            import httpx

            with open(pdf_path, "rb") as fh:
                resp = httpx.post(
                    f"{self.url}/api/processReferences",
                    files={"input": (Path(pdf_path).name, fh, "application/pdf")},
                    data={"consolidateCitations": "0"},
                    timeout=180.0,
                )
            resp.raise_for_status()
            return parse_tei_references(resp.text), resp.status_code, {}

        entry = self.cache.get_or_fetch(
            self.name, "references", cache_key_params, fetch, refresh=self.refresh
        )
        self._last_entry = entry
        return list(entry.body or [])

    # -- DOI resolution --------------------------------------------------

    def _match(self, ref: Mapping[str, Any]) -> tuple[str | None, float]:
        if ref.get("doi"):
            return str(ref["doi"]), 100.0
        if self.crossref is None or not ref.get("title"):
            return None, 0.0
        query_parts = [str(ref.get("title") or "")]
        if ref.get("authors"):
            query_parts.append(str(ref["authors"][0]))
        if ref.get("year"):
            query_parts.append(str(ref["year"]))
        self.match_stats["attempted"] += 1
        candidates = self.crossref.match_bibliographic(" ".join(query_parts))
        target = normalize_title(ref.get("title"))
        best: tuple[str | None, float] = (None, 0.0)
        for work, score in candidates:
            if score < self.min_match_score or not work.doi:
                continue
            sim = JaroWinkler.similarity(target, work.title_norm)
            if sim < self.min_title_similarity:
                continue
            ry, wy = ref.get("year"), work.year
            if ry and wy and abs(int(ry) - int(wy)) > 1:
                continue
            if sim > best[1]:
                best = (work.doi, sim)
        if best[0]:
            self.match_stats["matched"] += 1
        return best

    # -- interface -------------------------------------------------------

    def resolve(self, ident: str) -> Work | None:
        return None

    def references(self, work: Work) -> list[Work]:
        pdf_path = self.pdf_map.get(work.key)
        if not pdf_path or not Path(pdf_path).exists():
            return []
        out: list[Work] = []
        for ref in self._process_pdf(work.key, pdf_path):
            doi, confidence = self._match(ref)
            title = str(ref.get("title") or "")
            year = normalize_year(ref.get("year"))
            surname = ref["authors"][0] if ref.get("authors") else None
            if not title and not doi:
                continue
            if doi is None:
                self.match_stats["unresolved"] += 1
            out.append(
                Work(
                    key=canonical_key(
                        doi=doi, title=title, year=year, first_author_surname=surname
                    ),
                    title=title,
                    title_norm=normalize_title(title),
                    doi=doi,
                    year=year,
                    authors=tuple(Author(surname=a) for a in ref.get("authors") or []),
                    venue=ref.get("venue"),
                    unresolved=doi is None,
                    source_ids={"grobid_confidence": f"{confidence:.3f}"},
                )
            )
        return dedupe_preserving_order(out)

    def citations(self, work: Work) -> list[Work]:
        return []
