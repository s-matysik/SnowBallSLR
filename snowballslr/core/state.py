"""Persistent run state.

State is written after every committed phase, so a run interrupted mid-flight
resumes from the last consistent point rather than restarting. Serialization is
sorted throughout (INV-3).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import StateError
from ..types import Decision, Work
from .iteration import IterationHistory, IterationStats

__all__ = ["Phase", "RunState"]


class Phase:
    """Where the run currently sits in the iteration cycle."""

    READY = "ready"          # frontier available, next step() will expand
    AWAITING_LABELS = "awaiting_labels"
    STOPPED = "stopped"


@dataclass
class RunState:
    iteration: int = 0
    phase: str = Phase.READY
    works: dict[str, Work] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    # Ranked screening queue. Unlike every other collection this is stored in
    # ranked order, not key order: the order is the product. It is still
    # deterministic, because ranking breaks ties by canonical key.
    pending: list[str] = field(default_factory=list)
    pending_scores: dict[str, float] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)
    history: IterationHistory = field(default_factory=IterationHistory)
    stopped_by: str | None = None
    last_decisions: list[dict[str, Any]] = field(default_factory=list)

    # -- registry --------------------------------------------------------

    def upsert(self, work: Work, precedence: Sequence[str] = ()) -> Work:
        key = self.resolve_alias(work.key)
        existing = self.works.get(key)
        merged = existing.merged_with(work, precedence) if existing else work
        if merged.key != key:
            from dataclasses import replace

            merged = replace(merged, key=key)
        self.works[key] = merged
        return merged

    def resolve_alias(self, key: str) -> str:
        seen: set[str] = set()
        current = key
        while current in self.aliases and current not in seen:
            seen.add(current)
            current = self.aliases[current]
        return current

    def apply_aliases(self, aliases: Mapping[str, str]) -> None:
        """Fold absorbed keys into their representative, once, at end of iteration."""
        for absorbed, representative in sorted(aliases.items()):
            if absorbed == representative:
                continue
            self.aliases[absorbed] = representative
            if absorbed in self.works:
                work = self.works.pop(absorbed)
                self.upsert(work)
            if absorbed in self.decisions:
                self.decisions.setdefault(representative, self.decisions.pop(absorbed))
            self.frontier = [representative if k == absorbed else k for k in self.frontier]
            self.expanded = [representative if k == absorbed else k for k in self.expanded]
            self.pending = [representative if k == absorbed else k for k in self.pending]
            if absorbed in self.pending_scores:
                self.pending_scores[representative] = self.pending_scores.pop(absorbed)
        self.frontier = sorted(set(self.frontier))
        self.expanded = sorted(set(self.expanded))
        seen: set[str] = set()
        self.pending = [k for k in self.pending if not (k in seen or seen.add(k))]

    # -- decisions -------------------------------------------------------

    def decide(self, key: str, decision: Decision) -> None:
        resolved = self.resolve_alias(key)
        if resolved not in self.works:
            raise StateError(f"unknown key in labels: {key}")
        self.decisions[resolved] = str(decision)

    def decision_for(self, key: str) -> Decision:
        return Decision(self.decisions.get(self.resolve_alias(key), Decision.UNSCREENED))

    @property
    def included(self) -> list[Work]:
        return [
            self.works[k]
            for k in sorted(self.decisions)
            if self.decisions[k] == Decision.INCLUDE and k in self.works
        ]

    @property
    def excluded(self) -> list[Work]:
        return [
            self.works[k]
            for k in sorted(self.decisions)
            if self.decisions[k] == Decision.EXCLUDE and k in self.works
        ]

    @property
    def included_keys(self) -> list[str]:
        return sorted(k for k, v in self.decisions.items() if v == Decision.INCLUDE)

    @property
    def screened_keys(self) -> list[str]:
        return sorted(
            k for k, v in self.decisions.items() if v in (Decision.INCLUDE, Decision.EXCLUDE)
        )

    def pending_works(self) -> list[Work]:
        """Pending candidates in ranked order (INV-3: order is explicit, not incidental)."""
        return [self.works[k] for k in self.pending if k in self.works]

    # -- frontier --------------------------------------------------------

    def enqueue(self, keys: Iterable[str]) -> None:
        already = set(self.expanded) | set(self.frontier)
        new = sorted({self.resolve_alias(k) for k in keys} - already)
        self.frontier = sorted(set(self.frontier) | set(new))

    def take_frontier(self) -> list[str]:
        batch = sorted(self.frontier)
        self.frontier = []
        self.expanded = sorted(set(self.expanded) | set(batch))
        return batch

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "phase": self.phase,
            "works": {k: self.works[k].to_dict() for k in sorted(self.works)},
            "decisions": dict(sorted(self.decisions.items())),
            "frontier": sorted(self.frontier),
            "expanded": sorted(self.expanded),
            "pending": list(self.pending),
            "pending_scores": dict(sorted(self.pending_scores.items())),
            "aliases": dict(sorted(self.aliases.items())),
            "seeds": sorted(self.seeds),
            "history": self.history.to_dict(),
            "stopped_by": self.stopped_by,
            "last_decisions": self.last_decisions,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> RunState:
        p = Path(path)
        if not p.exists():
            raise StateError(f"run state not found: {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        return RunState(
            iteration=int(d.get("iteration", 0)),
            phase=str(d.get("phase", Phase.READY)),
            works={k: Work.from_dict(v) for k, v in (d.get("works") or {}).items()},
            decisions=dict(d.get("decisions") or {}),
            frontier=list(d.get("frontier") or []),
            expanded=list(d.get("expanded") or []),
            pending=list(d.get("pending") or []),
            pending_scores=dict(d.get("pending_scores") or {}),
            aliases=dict(d.get("aliases") or {}),
            seeds=list(d.get("seeds") or []),
            history=IterationHistory.from_dict(d.get("history") or {}),
            stopped_by=d.get("stopped_by"),
            last_decisions=list(d.get("last_decisions") or []),
        )

    # -- convenience -----------------------------------------------------

    def record_iteration(self, stats: IterationStats) -> None:
        self.history.append(stats)
