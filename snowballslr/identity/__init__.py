"""Record identity: normalization, canonical keys, deduplication."""

from .dedup import DedupResult, MergeEvent, blocking_key, deduplicate
from .keys import canonical_key, parse_identifier, signature_key
from .normalize import (
    normalize_doi,
    normalize_surname,
    normalize_title,
    normalize_year,
    tokenize,
)

__all__ = [
    "DedupResult",
    "MergeEvent",
    "blocking_key",
    "canonical_key",
    "deduplicate",
    "normalize_doi",
    "normalize_surname",
    "normalize_title",
    "normalize_year",
    "parse_identifier",
    "signature_key",
    "tokenize",
]
