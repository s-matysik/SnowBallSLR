"""Per-iteration bookkeeping consumed by the stopping rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["IterationHistory", "IterationStats"]


@dataclass(frozen=True, slots=True)
class IterationStats:
    n: int
    n_expanded: int
    n_raw: int
    n_after_dedup: int
    n_eligible: int
    n_screened: int
    n_included: int
    n_excluded: int
    cumulative_screened: int
    cumulative_included: int
    frontier_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_expanded": self.n_expanded,
            "n_raw": self.n_raw,
            "n_after_dedup": self.n_after_dedup,
            "n_eligible": self.n_eligible,
            "n_screened": self.n_screened,
            "n_included": self.n_included,
            "n_excluded": self.n_excluded,
            "cumulative_screened": self.cumulative_screened,
            "cumulative_included": self.cumulative_included,
            "frontier_size": self.frontier_size,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> IterationStats:
        return IterationStats(**{k: int(v) for k, v in d.items()})

    @property
    def yield_rate(self) -> float:
        return self.n_included / self.n_screened if self.n_screened else 0.0


@dataclass(frozen=True, slots=True)
class RecallEstimateRecord:
    """The part of a recall estimate the stopping rules need across iterations.

    Only the fields required to judge closure are retained, so the trail stays
    small and serializes cleanly into ``state.json``.
    """

    n: int
    estimable: bool
    n_hat: float | None = None
    n_observed: int | None = None
    recall: float | None = None
    recall_lower: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "estimable": self.estimable,
            "n_hat": self.n_hat,
            "n_observed": self.n_observed,
            "recall": self.recall,
            "recall_lower": self.recall_lower,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> RecallEstimateRecord:
        return RecallEstimateRecord(
            n=int(d["n"]),
            estimable=bool(d.get("estimable")),
            n_hat=d.get("n_hat"),
            n_observed=d.get("n_observed"),
            recall=d.get("recall"),
            recall_lower=d.get("recall_lower"),
        )


@dataclass
class IterationHistory:
    stats: list[IterationStats] = field(default_factory=list)
    recall_estimate: Any = None  # RecallEstimate | None, set by the run before evaluation
    # One record per iteration at which a recall estimate was attempted. The
    # closure guard in the estimated_recall rule reads this to decide whether
    # N-hat has settled; it must therefore survive save/load.
    recall_estimate_trail: list[RecallEstimateRecord] = field(default_factory=list)

    def append(self, s: IterationStats) -> None:
        self.stats = [x for x in self.stats if x.n != s.n]
        self.stats.append(s)
        self.stats.sort(key=lambda x: x.n)

    def record_estimate(self, n: int, estimate: Any) -> None:
        """Attach the iteration's recall estimate and extend the trail."""
        self.recall_estimate = estimate
        if estimate is None:
            return
        rec = RecallEstimateRecord(
            n=n,
            estimable=bool(getattr(estimate, "estimable", False)),
            n_hat=getattr(estimate, "n_hat", None),
            n_observed=getattr(estimate, "n_observed", None),
            recall=getattr(estimate, "recall", None),
            recall_lower=getattr(estimate, "recall_lower", None),
        )
        self.recall_estimate_trail = [
            x for x in self.recall_estimate_trail if x.n != n
        ] + [rec]
        self.recall_estimate_trail.sort(key=lambda x: x.n)

    def __len__(self) -> int:
        return len(self.stats)

    def __iter__(self) -> Iterator[IterationStats]:
        return iter(self.stats)

    def __getitem__(self, i: int) -> IterationStats:
        return self.stats[i]

    @property
    def last(self) -> IterationStats | None:
        return self.stats[-1] if self.stats else None

    @property
    def cumulative_included(self) -> list[int]:
        return [s.cumulative_included for s in self.stats]

    @property
    def cumulative_screened(self) -> int:
        return self.stats[-1].cumulative_screened if self.stats else 0

    @property
    def total_included(self) -> int:
        return self.stats[-1].cumulative_included if self.stats else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": [s.to_dict() for s in self.stats],
            "recall_estimate_trail": [r.to_dict() for r in self.recall_estimate_trail],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> IterationHistory:
        h = IterationHistory()
        for s in d.get("stats") or []:
            h.append(IterationStats.from_dict(s))
        h.recall_estimate_trail = sorted(
            (RecallEstimateRecord.from_dict(r) for r in (d.get("recall_estimate_trail") or [])),
            key=lambda x: x.n,
        )
        return h
