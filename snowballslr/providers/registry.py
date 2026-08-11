"""Provider construction from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import Config
from ..determinism.cache import Cache
from ..errors import ConfigError
from .base import BaseProvider, RetryPolicy
from .crossref import CrossrefProvider
from .grobid import GrobidProvider
from .offline import OfflineProvider
from .openalex import OpenAlexProvider
from .semanticscholar import SemanticScholarProvider

__all__ = ["build_providers", "provider_descriptors"]


def build_providers(
    config: Config,
    cache: Cache,
    *,
    offline_graph: Mapping[str, Any] | str | None = None,
) -> list[BaseProvider]:
    """Instantiate providers in the configured order."""
    retry = RetryPolicy(
        max_retries=config.determinism.max_retries,
        base=config.determinism.backoff_base,
        factor=config.determinism.backoff_factor,
    )
    common = {"retry": retry, "refresh": config.determinism.refresh}

    if offline_graph is not None:
        return [OfflineProvider(cache, graph=offline_graph, **common)]

    built: dict[str, BaseProvider] = {}
    for name in config.providers.order:
        if name == "openalex":
            s = config.providers.openalex
            built[name] = OpenAlexProvider(
                cache,
                mailto=s.mailto,
                base_url=s.base_url,
                per_page=s.per_page,
                rps=s.rps,
                **common,
            )
        elif name == "crossref":
            s = config.providers.crossref
            built[name] = CrossrefProvider(
                cache, mailto=s.mailto, base_url=s.base_url, rps=s.rps, **common
            )
        elif name == "semanticscholar":
            s = config.providers.semanticscholar
            built[name] = SemanticScholarProvider(
                cache,
                api_key_env=s.api_key_env,
                base_url=s.base_url,
                rps=s.rps,
                **common,
            )
        elif name == "grobid":
            s = config.providers.grobid
            built[name] = GrobidProvider(
                cache,
                url=s.url,
                pdf_map=s.pdf_map,
                crossref=built.get("crossref"),
                min_match_score=s.min_match_score,
                min_title_similarity=s.min_title_similarity,
                **common,
            )
        else:
            raise ConfigError(f"unknown provider: {name}")
    return [built[n] for n in config.providers.order]


def provider_descriptors(providers: list[BaseProvider]) -> list[dict[str, Any]]:
    """Manifest-ready provider descriptions (no secrets)."""
    out: list[dict[str, Any]] = []
    for p in providers:
        out.append(
            {
                "name": p.name,
                "base_url": getattr(p, "base_url", getattr(p, "url", "")),
                "supports": sorted(str(d) for d in p.supports),
                "polite_pool": bool(getattr(p, "mailto", None)),
            }
        )
    return out
