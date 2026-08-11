"""Run manifest (spec 9.2).

The manifest is the contract that ``verify`` checks against. It records the
resolved configuration, per-iteration hashes and every output artifact hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import canonical_json

__all__ = ["IterationRecord", "RunManifest", "hash_file", "hash_obj"]


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def hash_obj(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class IterationRecord:
    n: int
    frontier_hash: str
    candidates_hash: str | None = None
    labels_hash: str | None = None
    n_expanded: int = 0
    n_raw: int = 0
    n_after_dedup: int = 0
    n_eligible: int = 0
    n_screened: int = 0
    n_included: int = 0
    n_excluded: int = 0
    stop_decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "frontier_hash": self.frontier_hash,
            "candidates_hash": self.candidates_hash,
            "labels_hash": self.labels_hash,
            "n_expanded": self.n_expanded,
            "n_raw": self.n_raw,
            "n_after_dedup": self.n_after_dedup,
            "n_eligible": self.n_eligible,
            "n_screened": self.n_screened,
            "n_included": self.n_included,
            "n_excluded": self.n_excluded,
            "stop_decisions": self.stop_decisions,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> IterationRecord:
        return IterationRecord(
            n=int(d["n"]),
            frontier_hash=str(d["frontier_hash"]),
            candidates_hash=d.get("candidates_hash"),
            labels_hash=d.get("labels_hash"),
            n_expanded=int(d.get("n_expanded", 0)),
            n_raw=int(d.get("n_raw", 0)),
            n_after_dedup=int(d.get("n_after_dedup", 0)),
            n_eligible=int(d.get("n_eligible", 0)),
            n_screened=int(d.get("n_screened", 0)),
            n_included=int(d.get("n_included", 0)),
            n_excluded=int(d.get("n_excluded", 0)),
            stop_decisions=list(d.get("stop_decisions") or []),
        )


@dataclass
class RunManifest:
    snowballslr_version: str
    python_version: str
    config_hash: str
    config: dict[str, Any]
    providers: list[dict[str, Any]] = field(default_factory=list)
    ranker: dict[str, Any] = field(default_factory=dict)
    embedding_model: dict[str, Any] | None = None
    seeds: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    iterations: list[IterationRecord] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    cache_entries: int = 0
    stopped_by: str | None = None

    # -- construction ----------------------------------------------------

    @staticmethod
    def new(
        *,
        version: str,
        config: Any,
        seeds: Sequence[str],
        providers: Sequence[dict[str, Any]],
        ranker: Mapping[str, Any],
        started_at: str,
    ) -> RunManifest:
        return RunManifest(
            snowballslr_version=version,
            python_version=platform.python_version(),
            config_hash=config.config_hash,
            config=config.hashable(),
            providers=[dict(sorted(p.items())) for p in providers],
            ranker=dict(sorted(ranker.items())),
            seeds=sorted(seeds),
            started_at=started_at,
        )

    # -- io --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "snowballslr_version": self.snowballslr_version,
            "python_version": self.python_version,
            "config_hash": self.config_hash,
            "config": self.config,
            "providers": self.providers,
            "ranker": self.ranker,
            "embedding_model": self.embedding_model,
            "seeds": self.seeds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "iterations": [i.to_dict() for i in self.iterations],
            "outputs": dict(sorted(self.outputs.items())),
            "cache_entries": self.cache_entries,
            "stopped_by": self.stopped_by,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> RunManifest:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = RunManifest(
            snowballslr_version=str(d["snowballslr_version"]),
            python_version=str(d["python_version"]),
            config_hash=str(d["config_hash"]),
            config=dict(d.get("config") or {}),
            providers=list(d.get("providers") or []),
            ranker=dict(d.get("ranker") or {}),
            embedding_model=d.get("embedding_model"),
            seeds=list(d.get("seeds") or []),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            iterations=[IterationRecord.from_dict(i) for i in d.get("iterations") or []],
            outputs=dict(d.get("outputs") or {}),
            cache_entries=int(d.get("cache_entries", 0)),
            stopped_by=d.get("stopped_by"),
        )
        return m

    # -- mutation --------------------------------------------------------

    def record_iteration(self, record: IterationRecord) -> None:
        self.iterations = [i for i in self.iterations if i.n != record.n]
        self.iterations.append(record)
        self.iterations.sort(key=lambda i: i.n)

    def record_output(self, name: str, path: str | Path) -> None:
        self.outputs[name] = hash_file(path)
