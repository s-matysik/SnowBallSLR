"""Candidate ranking. Scores order the screening queue; they never decide inclusion."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import RankingSettings
from ..errors import ConfigError
from .base import Ranker, minmax
from .fusion import RRFFusion, rank_positions
from .lexical import BM25Ranker
from .network import NetworkRanker

__all__ = [
    "BM25Ranker",
    "NetworkRanker",
    "NullRanker",
    "RRFFusion",
    "Ranker",
    "build_ranker",
    "minmax",
    "rank_positions",
]


class NullRanker:
    """Deterministic no-op: everything scores zero, order falls back to key."""

    name = "none"

    def fit(self, included: Sequence[object]) -> None:
        return None

    def score(self, candidates: Sequence[object]):
        return {c.key: 0.0 for c in candidates}  # type: ignore[attr-defined]


def _component(name: str, settings: RankingSettings):
    if name == "bm25":
        return BM25Ranker()
    if name == "network":
        return NetworkRanker(settings.network_weights)
    if name == "embedding":
        from .embedding import EmbeddingRanker

        if not settings.embedding_model:
            raise ConfigError("embedding component requires ranking.embedding_model")
        return EmbeddingRanker(settings.embedding_model, settings.embedding_revision)
    raise ConfigError(f"unknown ranking component: {name}")


def build_ranker(settings: RankingSettings):
    if settings.type == "none":
        return NullRanker()
    if settings.type == "rrf":
        return RRFFusion(
            [_component(c, settings) for c in settings.components], k=settings.rrf_k
        )
    return _component(settings.type, settings)
