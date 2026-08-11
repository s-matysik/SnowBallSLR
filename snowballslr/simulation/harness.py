"""Multi-review, multi-seed experiment driver.

Every run executes against a frozen citation-graph snapshot, so a full benchmark
is offline, fast and exactly replicable -- and exercises the same determinism
layer the library ships for users.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..core.run import Run
from .metrics import (
    calibration,
    evaluate_stopping_rules,
    recall_curve,
    screening_burden,
)
from .oracle import Oracle

__all__ = ["ReviewSpec", "SeedStrategy", "TrialResult", "run_benchmark", "run_trial"]


class SeedStrategy:
    RANDOM = "random"
    MOST_CITED = "most_cited"
    OLDEST = "oldest"
    BOOLEAN = "boolean"

    ALL = (RANDOM, MOST_CITED, OLDEST, BOOLEAN)


@dataclass
class ReviewSpec:
    """One benchmark review: a frozen graph plus its known inclusion list."""

    name: str
    graph: dict[str, Any]
    gold: frozenset[str]
    boolean_seeds: tuple[str, ...] = ()
    domain: str = "unknown"

    @staticmethod
    def from_files(
        name: str,
        graph_path: str | Path,
        gold_path: str | Path,
        *,
        domain: str = "unknown",
        boolean_seeds: Sequence[str] = (),
    ) -> ReviewSpec:
        from .oracle import load_gold_standard

        return ReviewSpec(
            name=name,
            graph=json.loads(Path(graph_path).read_text(encoding="utf-8")),
            gold=frozenset(load_gold_standard(gold_path)),
            boolean_seeds=tuple(boolean_seeds),
            domain=domain,
        )


def select_seeds(
    spec: ReviewSpec, strategy: str, k: int, rng: random.Random
) -> list[str]:
    """Pick a start set. Strategy matters more than size, so all four are tested."""
    gold = sorted(spec.gold)
    works = spec.graph.get("works") or {}

    if strategy == SeedStrategy.BOOLEAN and spec.boolean_seeds:
        pool = [s for s in spec.boolean_seeds if s in spec.gold] or gold
        return sorted(pool[:k])
    if strategy == SeedStrategy.MOST_CITED:
        ranked = sorted(
            gold,
            key=lambda key: (-int((works.get(key) or {}).get("cited_by_count") or 0), key),
        )
        return sorted(ranked[:k])
    if strategy == SeedStrategy.OLDEST:
        ranked = sorted(
            gold, key=lambda key: (int((works.get(key) or {}).get("year") or 9999), key)
        )
        return sorted(ranked[:k])
    return sorted(rng.sample(gold, min(k, len(gold))))


@dataclass
class TrialResult:
    review: str
    domain: str
    strategy: str
    seed_size: int
    repeat: int
    seeds: list[str]
    n_gold: int
    n_found: int
    final_recall: float
    iterations: int
    total_screened: int
    stopped_by: str | None
    burden_80: int | None
    burden_90: int | None
    burden_95: int | None
    estimated_recall: float | None
    stopping_outcomes: list[dict[str, Any]] = field(default_factory=list)
    curve: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review": self.review,
            "domain": self.domain,
            "strategy": self.strategy,
            "seed_size": self.seed_size,
            "repeat": self.repeat,
            "seeds": self.seeds,
            "n_gold": self.n_gold,
            "n_found": self.n_found,
            "final_recall": self.final_recall,
            "iterations": self.iterations,
            "total_screened": self.total_screened,
            "stopped_by": self.stopped_by,
            "burden_80": self.burden_80,
            "burden_90": self.burden_90,
            "burden_95": self.burden_95,
            "estimated_recall": self.estimated_recall,
            "stopping_outcomes": self.stopping_outcomes,
            "curve": self.curve,
        }


def run_trial(
    spec: ReviewSpec,
    strategy: str,
    seed_size: int,
    repeat: int,
    workdir: str | Path,
    config: Config | None = None,
    *,
    target_recall: float = 0.95,
) -> TrialResult:
    """One (review, strategy, seed size, repeat) cell of the design."""
    rng = random.Random(f"{spec.name}|{strategy}|{seed_size}|{repeat}")
    seeds = select_seeds(spec, strategy, seed_size, rng)
    oracle = Oracle.from_keys(spec.gold)

    cfg = config or Config()
    root = Path(workdir) / f"{spec.name}_{strategy}_{seed_size}_{repeat}"
    run = Run.init(root, seeds, cfg, offline_graph=spec.graph)

    decisions_by_iteration: dict[int, list[dict[str, Any]]] = {}
    while not run.stopped and run.state.iteration < cfg.run.max_iterations:
        candidates = run.step()
        if not candidates:
            break
        run.label({w.key: oracle.decide(w.key) for w in candidates})
        decisions_by_iteration[run.state.iteration] = list(run.state.last_decisions)

    found = set(run.state.included_keys) & spec.gold
    curve = recall_curve(list(run.state.history), len(spec.gold))
    outcomes = evaluate_stopping_rules(
        curve, decisions_by_iteration, target_recall=target_recall, n_gold=len(spec.gold)
    )
    est = run.estimate_recall()
    run.close()

    last = run.state.history.last
    return TrialResult(
        review=spec.name,
        domain=spec.domain,
        strategy=strategy,
        seed_size=seed_size,
        repeat=repeat,
        seeds=seeds,
        n_gold=len(spec.gold),
        n_found=len(found),
        final_recall=len(found) / len(spec.gold) if spec.gold else 0.0,
        iterations=len(run.state.history),
        total_screened=last.cumulative_screened if last else 0,
        stopped_by=run.state.stopped_by,
        burden_80=screening_burden(curve, 0.80),
        burden_90=screening_burden(curve, 0.90),
        burden_95=screening_burden(curve, 0.95),
        estimated_recall=est.recall if est.estimable else None,
        stopping_outcomes=[o.to_dict() for o in outcomes],
        curve=curve.to_dict(),
    )


def run_benchmark(
    specs: Sequence[ReviewSpec],
    workdir: str | Path,
    *,
    strategies: Sequence[str] = SeedStrategy.ALL,
    seed_sizes: Sequence[int] = (1, 3, 5, 10),
    repeats: int = 10,
    config: Config | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Full factorial benchmark. Variance across seed draws is reported, not hidden."""
    results: list[TrialResult] = []
    for spec in specs:
        for strategy in strategies:
            for size in seed_sizes:
                # Deterministic strategies need only one repeat.
                n_repeats = repeats if strategy == SeedStrategy.RANDOM else 1
                for r in range(n_repeats):
                    results.append(
                        run_trial(spec, strategy, size, r, workdir, config)
                    )

    estimated = [t.estimated_recall for t in results]
    actual = [t.final_recall for t in results]
    summary = {
        "n_trials": len(results),
        "n_reviews": len(specs),
        "mean_final_recall": (
            sum(t.final_recall for t in results) / len(results) if results else 0.0
        ),
        "calibration": calibration(
            [e for e in estimated if e is not None],
            [a for e, a in zip(estimated, actual, strict=False) if e is not None],
        ),
        "results": [t.to_dict() for t in results],
    }
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary
