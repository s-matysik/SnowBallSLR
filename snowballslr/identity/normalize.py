"""Deterministic normalization of bibliographic fields.

Every function here is idempotent: ``f(f(x)) == f(x)``. This is asserted by
property tests and is what makes canonical keys stable across providers.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "STOPWORDS",
    "normalize_doi",
    "normalize_surname",
    "normalize_title",
    "normalize_year",
    "strip_correction_prefix",
    "tokenize",
]

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
    "doi ",
)

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

_CORRECTION_PREFIXES = (
    "erratum:",
    "erratum to:",
    "correction:",
    "correction to:",
    "corrigendum:",
    "corrigendum to:",
    "retraction:",
    "retraction note:",
    "withdrawn:",
)


def normalize_doi(raw: str | None) -> str | None:
    """Return a bare, lowercase DOI (``10.xxxx/yyy``) or ``None`` if invalid."""
    if not raw:
        return None
    s = str(raw).strip()
    low = s.lower()
    for prefix in _DOI_PREFIXES:
        if low.startswith(prefix):
            s = s[len(prefix) :]
            low = s.lower()
            break
    s = s.strip().strip("<>[]() ")
    # Trailing sentence punctuation is a common artifact of reference parsing.
    s = re.sub(r"[.,;]+$", "", s)
    s = s.lower()
    if not s:
        return None
    return s if _DOI_RE.match(s) else None


def strip_correction_prefix(title: str) -> tuple[str, bool]:
    """Strip an erratum/correction prefix. Returns ``(title, was_correction)``."""
    low = title.strip().lower()
    for prefix in _CORRECTION_PREFIXES:
        if low.startswith(prefix):
            return title.strip()[len(prefix) :].strip(), True
    return title.strip(), False


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


def normalize_title(raw: str | None) -> str:
    """NFKD fold, strip correction prefixes, drop non-alphanumerics, collapse space.

    Folding happens *before* prefix stripping so that accented spellings of the
    same prefix normalize identically.
    """
    if not raw:
        return ""
    folded = _fold(str(raw)).strip()
    for prefix in _CORRECTION_PREFIXES:
        if folded.startswith(prefix):
            folded = folded[len(prefix) :].strip()
            break
    cleaned = re.sub(r"[^0-9a-z]+", " ", folded)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_surname(raw: str | None) -> str:
    """Fold to lowercase alphabetic characters only."""
    if not raw:
        return ""
    folded = _fold(str(raw))
    return re.sub(r"[^a-z]+", "", folded)


def normalize_year(raw: object) -> int | None:
    """Extract the earliest plausible 4-digit year from a value."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if 1000 <= raw <= 2999 else None
    matches = re.findall(r"(1[0-9]{3}|2[0-9]{3})", str(raw))
    if not matches:
        return None
    return min(int(m) for m in matches)


# -- lexical tokenization (used by the BM25 ranker) -----------------------

STOPWORDS: frozenset[str] = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]
)


def tokenize(text: str | None) -> list[str]:
    """Deterministic tokenizer: normalized title pipeline, stopwords dropped."""
    if not text:
        return []
    return [
        tok
        for tok in normalize_title(text).split()
        if len(tok) > 1 and tok not in STOPWORDS
    ]
