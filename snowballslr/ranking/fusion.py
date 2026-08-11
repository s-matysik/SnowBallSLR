"""Reciprocal rank fusion.

Chosen as the default because it is deterministic, training-free and trivial to
justify in review: no learned weights, no seeds, no drift between runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..types import Work

__all__ = ["RRFFusion", "rank_positions"]


def rank_positions(scores: Mapping[str, float]) -> dict[str, int]:
    """1-based ranks, ties broken by canonical key ascending (INV-3)."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {key: i + 1 for i, (key, _) in enumerate(ordered)}


class RRFFusion:
    name = "rrf"

    def __init__(self, rankers: Sequence[object], k: int = 60) -> None:
        if not rankers:
            raise ValueError("RRFFusion requires at least one component ranker")
        self.rankers = list(rankers)
        self.k = k

    def fit(self, included: Sequence[Work]) -> None:
        for r in self.rankers:
            r.fit(included)  # type: ignore[attr-defined]

    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]:
        keys = sorted(w.key for w in candidates)
        totals = {k: 0.0 for k in keys}
        for r in self.rankers:
            ranks = rank_positions(dict(r.score(candidates)))  # type: ignore[attr-defined]
            for key in keys:
                totals[key] += 1.0 / (self.k + ranks.get(key, len(keys) + 1))
        return totals
