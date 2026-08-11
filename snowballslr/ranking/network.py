"""Citation-graph ranker.

Works entirely from graph position, so it still ranks usefully when abstracts
are missing -- which is the common case for records reached through reference
lists. No competing snowballing tool prioritizes candidates this way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..types import Direction, Work
from .base import minmax

__all__ = ["NetworkRanker"]


class NetworkRanker:
    name = "network"

    def __init__(self, weights: Sequence[float] = (0.4, 0.25, 0.25, 0.10)) -> None:
        if len(weights) != 4:
            raise ValueError("network ranker expects exactly 4 weights")
        self.weights = tuple(float(w) for w in weights)
        self._included_keys: frozenset[str] = frozenset()
        self._included_refs: dict[str, frozenset[str]] = {}

    def fit(self, included: Sequence[Work]) -> None:
        self._included_keys = frozenset(w.key for w in included)
        self._included_refs = {
            w.key: frozenset(w.parent_keys) for w in sorted(included, key=lambda x: x.key)
        }

    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]:
        ordered = sorted(candidates, key=lambda w: w.key)
        n_parents: dict[str, float] = {}
        cocitation: dict[str, float] = {}
        coupling: dict[str, float] = {}
        direction_bonus: dict[str, float] = {}

        for w in ordered:
            parents = set(w.parent_keys) & self._included_keys
            n_parents[w.key] = float(len(parents))
            # Co-citation: included works that cite (or are cited by) the same parents.
            co = 0
            for other_key, refs in self._included_refs.items():
                if other_key in parents:
                    continue
                if refs & parents:
                    co += 1
            cocitation[w.key] = float(co)
            coupling[w.key] = float(len(set(w.parent_keys) & set(self._included_refs)))
            dirs = set(w.directions)
            direction_bonus[w.key] = (
                1.0
                if {Direction.BACKWARD, Direction.FORWARD} <= dirs
                else 0.0
            )

        np_, cc, cp, db = (
            minmax(n_parents),
            minmax(cocitation),
            minmax(coupling),
            direction_bonus,
        )
        w0, w1, w2, w3 = self.weights
        return {
            k: w0 * np_.get(k, 0.0)
            + w1 * cc.get(k, 0.0)
            + w2 * cp.get(k, 0.0)
            + w3 * db.get(k, 0.0)
            for k in sorted(n_parents)
        }
