"""Content-addressed cache for provider responses (spec 9.1).

Raw bodies are stored unmodified, so a parser bugfix never forces a refetch.
Cache keys are derived from ``provider|method|canonical_json(params)`` -- the
same request always maps to the same path, on any machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import canonical_json
from ..errors import OfflineError
from ..types import utcnow

__all__ = ["Cache", "CacheEntry", "cache_key", "sha256_of"]


def sha256_of(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return "sha256:" + hashlib.sha256(data).hexdigest()


def cache_key(provider: str, method: str, params: Mapping[str, Any]) -> str:
    material = f"{provider}|{method}|{canonical_json(dict(params))}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    provider: str
    method: str
    params: dict[str, Any]
    retrieved_at: str
    http_status: int
    response_hash: str
    provider_version: dict[str, Any]
    body: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "provider": self.provider,
            "method": self.method,
            "params": self.params,
            "retrieved_at": self.retrieved_at,
            "http_status": self.http_status,
            "response_hash": self.response_hash,
            "provider_version": self.provider_version,
            "body": self.body,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> CacheEntry:
        return CacheEntry(
            cache_key=str(d["cache_key"]),
            provider=str(d["provider"]),
            method=str(d["method"]),
            params=dict(d.get("params") or {}),
            retrieved_at=str(d["retrieved_at"]),
            http_status=int(d.get("http_status", 200)),
            response_hash=str(d["response_hash"]),
            provider_version=dict(d.get("provider_version") or {}),
            body=d.get("body"),
        )


class Cache:
    """On-disk, content-addressed store. Never expires entries by age."""

    def __init__(self, root: str | Path, *, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # -- paths -----------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    # -- read/write ------------------------------------------------------

    def read(self, key: str) -> CacheEntry | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        return CacheEntry.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def write(self, entry: CacheEntry) -> None:
        p = self.path_for(entry.cache_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )

    # -- main entry point ------------------------------------------------

    def get_or_fetch(
        self,
        provider: str,
        method: str,
        params: Mapping[str, Any],
        fetch: Callable[[], tuple[Any, int, dict[str, Any]]],
        *,
        refresh: bool = False,
    ) -> CacheEntry:
        """Return a cached entry, fetching only when absent (or ``refresh``).

        ``fetch`` returns ``(body, http_status, provider_version)``.
        """
        key = cache_key(provider, method, params)
        if not refresh:
            existing = self.read(key)
            if existing is not None:
                self.hits += 1
                return existing
        if self.offline:
            raise OfflineError(
                f"offline mode: no cache entry for {provider}.{method} "
                f"params={canonical_json(dict(params))}"
            )
        body, status, version = fetch()
        self.misses += 1
        entry = CacheEntry(
            cache_key=key,
            provider=provider,
            method=method,
            params=dict(params),
            retrieved_at=utcnow().isoformat().replace("+00:00", "Z"),
            http_status=status,
            response_hash=sha256_of(canonical_json(body)),
            provider_version=dict(version),
            body=body,
        )
        self.write(entry)
        return entry

    # -- inspection ------------------------------------------------------

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.rglob("*.json"))

    def entries(self) -> Iterator[CacheEntry]:
        for key in self.keys():
            entry = self.read(key)
            if entry is not None:
                yield entry

    def __len__(self) -> int:
        return len(self.keys())
