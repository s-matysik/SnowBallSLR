"""Provider protocol, rate limiting and retry policy (spec 4).

Backoff is deliberately *jitter-free*: retry timing must not introduce a
non-deterministic ordering of cache writes.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from ..determinism.cache import Cache
from ..errors import ProviderError, RateLimitError
from ..types import Direction, Work

__all__ = ["BaseProvider", "FetchResult", "Provider", "RateLimiter", "RetryPolicy"]

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FetchResult:
    body: Any
    status: int
    version: dict[str, Any]


class RateLimiter:
    """Token bucket. Sleeping affects wall clock only, never artifact content."""

    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 5
    base: float = 1.0
    factor: float = 2.0

    def delay(self, attempt: int) -> float:
        return self.base * (self.factor**attempt)


@runtime_checkable
class Provider(Protocol):
    name: str
    supports: frozenset[Direction]

    def resolve(self, ident: str) -> Work | None: ...
    def references(self, work: Work) -> list[Work]: ...
    def citations(self, work: Work) -> list[Work]: ...


class BaseProvider:
    """Shared HTTP plumbing: cache lookup, rate limiting, bounded retries."""

    name: str = "base"
    supports: frozenset[Direction] = frozenset()

    def __init__(
        self,
        cache: Cache,
        *,
        rps: float = 8.0,
        retry: RetryPolicy | None = None,
        client: httpx.Client | None = None,
        refresh: bool = False,
        iteration: int = 0,
    ) -> None:
        self.cache = cache
        self.limiter = RateLimiter(rps)
        self.retry = retry or RetryPolicy()
        self._client = client
        self.refresh = refresh
        self.iteration = iteration

    # -- http ------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            # follow_redirects is load-bearing, not a convenience. Crossref answers
            # 301 for DOIs whose registrant prefix has been reassigned (publisher
            # acquisitions, journal transfers) with an empty body; httpx does not
            # follow redirects by default, so such a record used to abort the whole
            # iteration. Redirect targets stay inside the same API host and the
            # cache is keyed on the *requested* parameters, so following them does
            # not affect determinism.
            self._client = httpx.Client(
                timeout=30.0, headers=self.headers(), follow_redirects=True
            )
        return self._client

    def headers(self) -> dict[str, str]:
        return {"User-Agent": "SnowBallSLR/1.0 (+https://github.com/s-matysik)"}

    def _http_get(self, url: str, params: dict[str, Any]) -> FetchResult:
        last_status = 0
        for attempt in range(self.retry.max_retries):
            self.limiter.wait()
            try:
                resp = self.client.get(url, params=params)
            except httpx.HTTPError as exc:  # pragma: no cover - network path
                if attempt == self.retry.max_retries - 1:
                    raise ProviderError(f"{self.name}: transport failure: {exc}") from exc
                time.sleep(self.retry.delay(attempt))
                continue
            last_status = resp.status_code
            if resp.status_code == 404:
                return FetchResult(body=None, status=404, version={})
            if resp.status_code in _RETRY_STATUS:
                if attempt == self.retry.max_retries - 1:
                    raise RateLimitError(
                        f"{self.name}: HTTP {resp.status_code} after "
                        f"{self.retry.max_retries} attempts"
                    )
                time.sleep(self.retry.delay(attempt))
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    f"{self.name}: HTTP {resp.status_code} for {url} "
                    f"params={params} :: {_error_detail(resp)}"
                )
            # A 2xx status does not guarantee a JSON body. Interception proxies,
            # CDN maintenance pages and WAF challenges all return HTML with
            # status 200, and an unguarded resp.json() surfaces that as a bare
            # JSONDecodeError from deep inside the cache callback -- losing the
            # whole iteration to an error that names neither the provider nor
            # the URL. Fail as a ProviderError that identifies both.
            try:
                body = resp.json()
            except ValueError as exc:
                ctype = resp.headers.get("content-type", "unknown")
                snippet = resp.text[:200].replace("\n", " ").strip()
                raise ProviderError(
                    f"{self.name}: HTTP {resp.status_code} for {url} returned a "
                    f"non-JSON body (content-type={ctype!r}). This usually means an "
                    f"intercepting proxy, a maintenance page or a rate-limit "
                    f"challenge rather than a bad request. First 200 bytes: "
                    f"{snippet!r}"
                ) from exc
            return FetchResult(
                body=body,
                status=resp.status_code,
                version=self.version_from_response(resp),
            )
        raise ProviderError(f"{self.name}: exhausted retries (last status {last_status})")

    def version_from_response(self, resp: httpx.Response) -> dict[str, Any]:
        return {}

    # -- cached fetch ----------------------------------------------------

    def fetch(self, method: str, params: dict[str, Any], url: str) -> Any:
        """Cached GET.

        Keys prefixed with ``_`` participate in the cache key but are never sent
        to the API: they exist to disambiguate requests whose query strings are
        identical (cursor pages, ID batches) without polluting the request.
        """
        wire = {k: v for k, v in params.items() if not k.startswith("_")}
        entry = self.cache.get_or_fetch(
            self.name,
            method,
            params,
            lambda: _as_tuple(self._http_get(url, wire)),
            refresh=self.refresh,
        )
        self._last_entry = entry
        return entry.body

    @property
    def last_response_hash(self) -> str:
        return getattr(self, "_last_entry", None).response_hash if hasattr(self, "_last_entry") else ""

    @property
    def last_retrieved_at(self) -> str:
        return getattr(self, "_last_entry", None).retrieved_at if hasattr(self, "_last_entry") else ""

    # -- interface -------------------------------------------------------

    def resolve(self, ident: str) -> Work | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def references(self, work: Work) -> list[Work]:
        return []

    def citations(self, work: Work) -> list[Work]:
        return []

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _as_tuple(result: FetchResult) -> tuple[Any, int, dict[str, Any]]:
    return result.body, result.status, result.version


def _error_detail(resp: httpx.Response) -> str:
    """Best-effort extraction of a provider's own error message.

    Discarding the response body on a 4xx turns a one-line diagnosis into an
    afternoon of guessing, so it is always surfaced.
    """
    try:
        body = resp.json()
    except Exception:
        return resp.text[:300]
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            if body.get(key):
                return str(body[key])[:300]
    return str(body)[:300]


def dedupe_preserving_order(items: Iterable[Work]) -> list[Work]:
    """Order-stable dedupe by key, then sorted -- providers must not leak order."""
    seen: dict[str, Work] = {}
    for w in items:
        if w.key not in seen:
            seen[w.key] = w
    return [seen[k] for k in sorted(seen)]


def chunked(seq: Sequence[str], size: int) -> list[list[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]
