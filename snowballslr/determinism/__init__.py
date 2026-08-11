"""Determinism layer: content-addressed cache, run manifest, drift verification."""

from .cache import Cache, CacheEntry, cache_key, sha256_of
from .manifest import IterationRecord, RunManifest, hash_file, hash_obj
from .verify import DiffClass, VerifyReport, verify_refresh, verify_replay

__all__ = [
    "Cache",
    "CacheEntry",
    "DiffClass",
    "IterationRecord",
    "RunManifest",
    "VerifyReport",
    "cache_key",
    "hash_file",
    "hash_obj",
    "sha256_of",
    "verify_refresh",
    "verify_replay",
]
