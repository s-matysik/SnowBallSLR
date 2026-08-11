"""Canonical key derivation.

Precedence (spec 5.2):

1. ``doi:<normalized-doi>``
2. ``oa:<openalex-id>``
3. ``s2:<corpus-id>``
4. ``sig:<sha1(title_norm|year|first_author_surname)[:16]>``

Keys must be stable across providers and across runs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from .normalize import normalize_doi, normalize_surname, normalize_title, normalize_year

__all__ = ["canonical_key", "normalize_source_id", "parse_identifier", "signature_key"]

_OA_RE = re.compile(r"(W\d+)", re.IGNORECASE)


def normalize_source_id(namespace: str, value: str | None) -> str | None:
    """Normalize a provider-specific identifier to its bare form."""
    if not value:
        return None
    s = str(value).strip()
    if namespace == "openalex":
        m = _OA_RE.search(s)
        return m.group(1).upper() if m else None
    if namespace == "s2":
        return s.split("/")[-1].strip() or None
    if namespace == "pmid":
        digits = re.sub(r"\D", "", s)
        return digits or None
    return s or None


def signature_key(
    title: str | None, year: object, first_author_surname: str | None
) -> str:
    payload = "|".join(
        [
            normalize_title(title),
            str(normalize_year(year) or ""),
            normalize_surname(first_author_surname),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"sig:{digest}"


def canonical_key(
    *,
    doi: str | None = None,
    source_ids: Mapping[str, str] | None = None,
    title: str | None = None,
    year: object = None,
    authors: Sequence[object] | None = None,
    first_author_surname: str | None = None,
) -> str:
    """Derive the canonical key for a record."""
    norm_doi = normalize_doi(doi)
    if norm_doi:
        return f"doi:{norm_doi}"

    ids = dict(source_ids or {})
    oa = normalize_source_id("openalex", ids.get("openalex"))
    if oa:
        return f"oa:{oa}"
    s2 = normalize_source_id("s2", ids.get("s2"))
    if s2:
        return f"s2:{s2}"

    surname = first_author_surname
    if surname is None and authors:
        first = authors[0]
        surname = getattr(first, "surname", None)
        if surname is None and isinstance(first, Mapping):
            surname = first.get("surname")
    return signature_key(title, year, surname)


def parse_identifier(raw: str) -> tuple[str, str]:
    """Classify a user-supplied seed identifier.

    Returns ``(namespace, value)`` where namespace is one of
    ``doi``, ``openalex``, ``s2``, ``pmid``.
    """
    s = raw.strip()
    doi = normalize_doi(s)
    if doi:
        return "doi", doi
    if s.lower().startswith("doi:"):
        doi = normalize_doi(s)
        if doi:
            return "doi", doi
    if _OA_RE.fullmatch(s) or "openalex.org" in s.lower():
        oa = normalize_source_id("openalex", s)
        if oa:
            return "openalex", oa
    if s.lower().startswith("pmid:"):
        return "pmid", re.sub(r"\D", "", s)
    if s.isdigit():
        return "pmid", s
    if s.lower().startswith(("s2:", "corpusid:")):
        return "s2", s.split(":", 1)[1].strip()
    return "s2", s
