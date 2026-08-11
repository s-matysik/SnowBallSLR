"""Ranker protocol.

INV-5: rankers see ``included`` read-only and return scores. They have no
handle on run state and cannot influence inclusion -- ranking orders the
screening queue and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..types import Work

__all__ = ["Ranker", "minmax"]


@runtime_checkable
class Ranker(Protocol):
    name: str

    def fit(self, included: Sequence[Work]) -> None: ...
    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]: ...


def minmax(values: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]; constant input maps to all zeros."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}
