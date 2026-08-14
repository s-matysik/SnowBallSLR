"""Run orchestration -- the snowballing state machine.

One iteration is::

    expand frontier -> normalize -> dedup -> eligibility -> rank -> export
        -> [external screening] -> ingest labels -> evaluate stopping rules

``step()`` performs everything up to the export and then halts; ``label()``
ingests decisions and evaluates the rules. The two are separate because
screening is external by design (INV-5) -- the library never decides inclusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import Config
from ..determinism.cache import Cache
from ..determinism.manifest import IterationRecord, RunManifest, hash_obj
from ..errors import ConfigError, LabelError, StateError
from ..estimate import estimate_recall
from ..identity.dedup import deduplicate
from ..identity.keys import parse_identifier
from ..providers.base import BaseProvider
from ..providers.registry import build_providers, provider_descriptors
from ..ranking import build_ranker
from ..report.audit import AuditLog
from ..report.export import (
    read_labels,
    write_candidates_csv,
    write_labels_template,
    write_ris,
)
from ..stopping.compose import RuleSet, build_rules
from ..types import Decision, Direction, Provenance, Work, utcnow
from .iteration import IterationStats
from .state import Phase, RunState

__all__ = ["Run"]


def _rejection_report(
    rejected: Mapping[str, list[Work]], n_fresh: int
) -> dict[str, Any]:
    """Summarise what the eligibility filter removed, and which values it saw.

    Listing the *observed* values alongside the counts is what turns "most
    records were filtered out" into "your `types` list does not match the
    vocabulary this provider actually returns".
    """
    n_rejected = sum(len(v) for v in rejected.values())
    observed: dict[str, dict[str, int]] = {}
    for reason, works in rejected.items():
        if reason == "type_not_allowed":
            field = "type"
        elif reason == "language_not_allowed":
            field = "language"
        else:
            continue
        counts: dict[str, int] = {}
        for w in works:
            value = getattr(w, field) or "(none)"
            counts[value] = counts.get(value, 0) + 1
        observed[field] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "n_fresh": n_fresh,
        "n_rejected": n_rejected,
        "share_rejected": (n_rejected / n_fresh) if n_fresh else 0.0,
        "by_reason": {k: len(v) for k, v in sorted(rejected.items())},
        "observed_values": observed,
    }

_VERSION = "1.0.0"


class Run:
    """A resumable snowballing run rooted at a directory."""

    def __init__(
        self,
        root: str | Path,
        config: Config,
        *,
        offline_graph: Mapping[str, Any] | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.config.validate_consistency()
        self.offline_graph = offline_graph

        self.cache = Cache(
            self.root / config.determinism.cache_dir, offline=config.determinism.offline
        )
        self.providers: list[BaseProvider] = build_providers(
            config, self.cache, offline_graph=offline_graph
        )
        self.ranker = build_ranker(config.ranking)
        self.rules = RuleSet(build_rules(config.stopping), config.stopping.mode)
        self.audit = AuditLog(self.root / "audit.jsonl")

        self.state = RunState()
        self.manifest: RunManifest | None = None
        self.last_rejection_report: dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def init(
        cls,
        root: str | Path,
        seeds: Sequence[str],
        config: Config | None = None,
        *,
        offline_graph: Mapping[str, Any] | str | None = None,
    ) -> Run:
        run = cls(root, config or Config(), offline_graph=offline_graph)
        run.root.mkdir(parents=True, exist_ok=True)
        (run.root / "iterations").mkdir(exist_ok=True)
        (run.root / "outputs").mkdir(exist_ok=True)
        run.audit.clear()

        run.manifest = RunManifest.new(
            version=_VERSION,
            config=run.config,
            seeds=list(seeds),
            providers=provider_descriptors(run.providers),
            ranker={"type": run.config.ranking.type, "components": run.config.ranking.components},
            started_at=utcnow().isoformat().replace("+00:00", "Z"),
        )
        run._resolve_seeds(seeds)
        run._persist()
        return run

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        offline_graph: Mapping[str, Any] | str | None = None,
        config: Config | None = None,
    ) -> Run:
        root = Path(root)
        cfg_path = root / "config.yaml"
        cfg = config or (Config.from_yaml(cfg_path) if cfg_path.exists() else Config())
        run = cls(root, cfg, offline_graph=offline_graph)
        run.state = RunState.load(root / "state.json")
        manifest_path = root / "run.json"
        if manifest_path.exists():
            run.manifest = RunManifest.load(manifest_path)
        return run

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state.save(self.root / "state.json")
        self.config.to_yaml(self.root / "config.yaml")
        if self.manifest is not None:
            self.manifest.cache_entries = len(self.cache)
            self.manifest.stopped_by = self.state.stopped_by
            self.manifest.save(self.root / "run.json")

    # -- seeds -----------------------------------------------------------

    def _resolve_seeds(self, seeds: Sequence[str]) -> None:
        resolved: list[str] = []
        for raw in sorted(set(seeds)):
            _namespace, value = parse_identifier(raw)
            work = None
            for provider in self.providers:
                try:
                    work = provider.resolve(value)
                except Exception as exc:
                    self.audit.emit(0, "seed_resolve_failed", ident=raw, error=str(exc))
                    continue
                if work is not None:
                    break
            if work is None:
                self.audit.emit(0, "seed_unresolved", ident=raw)
                continue
            prov = Provenance(
                provider="seed",
                direction=Direction.SEED,
                parent_key=None,
                iteration=0,
                retrieved_at="1970-01-01T00:00:00Z",
                response_hash="sha256:seed",
            )
            stored = self.state.upsert(
                work.with_provenance([prov]), self.config.providers.precedence
            )
            self.state.decisions[stored.key] = str(Decision.INCLUDE)
            resolved.append(stored.key)
            self.audit.emit(0, "seed_resolved", ident=raw, key=stored.key)

        self.state.seeds = sorted(resolved)
        self.state.enqueue(resolved)
        self.state.phase = Phase.READY

    # -- iteration -------------------------------------------------------

    @property
    def stopped(self) -> bool:
        return self.state.phase == Phase.STOPPED

    def step(self) -> list[Work]:
        """Expand the frontier and return ranked candidates awaiting screening."""
        if self.stopped:
            return []
        if self.state.phase == Phase.AWAITING_LABELS:
            # Idempotent: return the pending batch rather than refetching.
            return self.state.pending_works()
        if self.state.iteration >= self.config.run.max_iterations:
            self._stop("max_iterations")
            return []

        iteration = self.state.iteration + 1
        batch = self.state.take_frontier()
        if not batch:
            self._finalize_iteration(iteration, [], {}, n_expanded=0, n_raw=0)
            return []

        raw = self._expand(batch, iteration)
        result = deduplicate(
            raw,
            jw_threshold=self.config.dedup.jw_threshold,
            max_cluster_size=self.config.dedup.max_cluster_size,
            precedence=self.config.providers.precedence,
        )
        for merge in result.merges:
            self.audit.emit(
                iteration,
                "dedup_merge",
                representative=merge.representative,
                members=list(merge.members),
                tier=merge.tier,
                score=merge.score,
                conflicts={k: list(v) for k, v in merge.conflicts.items()},
            )
        for cluster in result.suspicious_clusters:
            self.audit.emit(
                iteration, "dedup_cluster_rejected", members=list(cluster)
            )

        for work in result.works:
            self.state.upsert(work, self.config.providers.precedence)
        self.state.apply_aliases(result.aliases)

        known = set(self.state.decisions)
        fresh = [
            self.state.works[self.state.resolve_alias(w.key)]
            for w in result.works
            if self.state.resolve_alias(w.key) not in known
        ]
        fresh = sorted({w.key: w for w in fresh}.values(), key=lambda w: w.key)

        eligible: list[Work] = []
        rejected: dict[str, list[Work]] = {}
        for w in fresh:
            reason = self._ineligibility_reason(w)
            if reason is None:
                eligible.append(w)
            else:
                rejected.setdefault(reason, []).append(w)

        n_ineligible = sum(len(v) for v in rejected.values())
        self.last_rejection_report = _rejection_report(rejected, len(fresh))
        if n_ineligible:
            self.audit.emit(
                iteration,
                "ineligible",
                n=n_ineligible,
                n_fresh=len(fresh),
                by_reason={k: len(v) for k, v in sorted(rejected.items())},
                observed_values=self.last_rejection_report["observed_values"],
            )

        scores = self._rank(eligible)
        if self.config.ranking.top_k:
            ordered = sorted(eligible, key=lambda w: (-scores.get(w.key, 0.0), w.key))
            eligible = ordered[: self.config.ranking.top_k]

        ranked = sorted(eligible, key=lambda w: (-scores.get(w.key, 0.0), w.key))
        self.state.pending = [w.key for w in ranked]
        self.state.pending_scores = {w.key: scores.get(w.key, 0.0) for w in ranked}
        self.state.iteration = iteration
        self.state.phase = Phase.AWAITING_LABELS if eligible else Phase.READY

        self._export_candidates(iteration, ranked, scores)
        self._record_manifest_iteration(
            iteration,
            batch=batch,
            n_raw=len(raw),
            n_after_dedup=len(result.works),
            n_eligible=len(eligible),
        )
        self._persist()

        if not eligible:
            # Nothing to screen: close the iteration immediately.
            self._finalize_iteration(
                iteration, [], scores, n_expanded=len(batch), n_raw=len(raw)
            )
            return []
        return ranked

    def label(
        self, decisions: Mapping[str, str | Decision], *, allow_empty: bool = False
    ) -> None:
        """Ingest screening decisions and evaluate the stopping rules.

        Set ``allow_empty`` to accept a batch in which no record was screened.
        """
        if self.state.phase != Phase.AWAITING_LABELS:
            raise StateError(
                f"cannot label in phase '{self.state.phase}'; run step() first"
            )
        pending = set(self.state.pending)
        unknown = sorted(set(decisions) - {self.state.resolve_alias(k) for k in pending} - pending)
        if unknown:
            raise LabelError(f"labels reference unknown keys: {unknown[:10]}")

        n_decided = sum(
            1
            for k in decisions
            if str(decisions[k]).strip().lower()
            in (Decision.INCLUDE.value, Decision.EXCLUDE.value)
        )
        if pending and n_decided == 0 and not allow_empty:
            # Without this guard the run terminates by `exhaustion` -- the frontier
            # is empty because nothing was included -- and reports a completed
            # snowballing run that never screened a single record. The failure is
            # silent and the PRISMA counts look superficially valid, so it is
            # refused rather than warned about.
            raise LabelError(
                f"none of the {len(pending)} pending record(s) carry a decision. "
                "Every 'decision' cell is empty or 'unscreened', so no record can "
                "enter the frontier and the run would stop by exhaustion having "
                "screened nothing. Fill the 'decision' column with 'include' or "
                "'exclude', or pass allow_empty=True if an empty batch is genuinely "
                "intended."
            )

        n_included = 0
        for key in sorted(decisions):
            value = str(decisions[key]).strip().lower()
            if value not in {d.value for d in Decision}:
                raise LabelError(f"invalid decision '{value}' for key {key}")
            if value == Decision.UNSCREENED:
                continue
            self.state.decide(key, Decision(value))
            if value == Decision.INCLUDE:
                n_included += 1

        newly_included = [
            self.state.resolve_alias(k)
            for k in sorted(decisions)
            if str(decisions[k]).strip().lower() == Decision.INCLUDE
        ]
        self.state.enqueue(newly_included)
        self.state.pending = []
        self.state.pending_scores = {}

        iteration = self.state.iteration
        n_screened = sum(
            1
            for k in decisions
            if str(decisions[k]).strip().lower() in (Decision.INCLUDE, Decision.EXCLUDE)
        )
        self._finalize_iteration(
            iteration,
            newly_included,
            {},
            n_expanded=len(self.state.expanded),
            n_raw=0,
            n_screened=n_screened,
            n_included=n_included,
        )

    def label_from_file(self, path: str | Path, *, allow_empty: bool = False) -> None:
        labels = read_labels(path)
        self.label({k: v[0] for k, v in labels.items()}, allow_empty=allow_empty)
        if self.manifest and self.manifest.iterations:
            self.manifest.iterations[-1].labels_hash = hash_obj(
                {k: v[0] for k, v in sorted(labels.items())}
            )
            self._persist()

    def run_to_saturation(self, oracle: Mapping[str, str], max_iterations: int | None = None) -> None:
        """Simulation loop: an oracle answers every screening decision."""
        limit = max_iterations or self.config.run.max_iterations
        while not self.stopped and self.state.iteration < limit:
            candidates = self.step()
            if not candidates:
                break
            decisions = {
                w.key: oracle.get(w.key, Decision.EXCLUDE.value) for w in candidates
            }
            self.label(decisions)

    # -- internals -------------------------------------------------------

    def _expand(self, batch: Sequence[str], iteration: int) -> list[Work]:
        directions = [Direction(d) for d in self.config.run.directions]
        found: list[Work] = []
        for parent_key in batch:
            parent = self.state.works.get(parent_key)
            if parent is None:
                continue
            if parent.unresolved:
                # A record with no persistent identifier cannot be looked up in any
                # provider: expanding it would issue a title query whose match is
                # unverifiable and would silently graft a wrong neighbourhood onto
                # the graph. Such records are retained, screened and counted in
                # PRISMA, but are terminal nodes.
                self.audit.emit(
                    iteration,
                    "expand_skipped_unresolved",
                    key=parent_key,
                    title=parent.title,
                )
                continue
            for provider in self.providers:
                provider.iteration = iteration
                for direction in directions:
                    if direction not in provider.supports:
                        continue
                    method = (
                        provider.references if direction == Direction.BACKWARD
                        else provider.citations
                    )
                    try:
                        neighbours = method(parent)
                    except Exception as exc:
                        self.audit.emit(
                            iteration,
                            "expand_failed",
                            key=parent_key,
                            provider=provider.name,
                            direction=str(direction),
                            error=str(exc),
                        )
                        raise
                    prov = Provenance(
                        provider=provider.name,
                        direction=direction,
                        parent_key=parent_key,
                        iteration=iteration,
                        retrieved_at=provider.last_retrieved_at or "1970-01-01T00:00:00Z",
                        response_hash=provider.last_response_hash or "sha256:unknown",
                    )
                    for w in neighbours:
                        found.append(w.with_provenance([prov]))
                    self.audit.emit(
                        iteration,
                        "expand",
                        key=parent_key,
                        provider=provider.name,
                        direction=str(direction),
                        n_returned=len(neighbours),
                    )
        return found

    def _ineligibility_reason(self, work: Work) -> str | None:
        """Why a record was filtered out, or ``None`` if it passes.

        A reason rather than a boolean: a filter that silently removes most of
        the corpus is indistinguishable from a corpus that simply is small, and
        that failure mode is expensive to discover late.
        """
        e = self.config.eligibility
        if e.require_title and not work.title.strip():
            return "no_title"
        if e.require_abstract and not (work.abstract or "").strip():
            return "no_abstract"
        if e.exclude_retracted and work.is_retracted:
            return "retracted"
        strict = not e.keep_unknown_metadata

        if work.year is None:
            if strict and (e.year_min is not None or e.year_max is not None):
                return "no_year"
        else:
            if e.year_min is not None and work.year < e.year_min:
                return "year_below_min"
            if e.year_max is not None and work.year > e.year_max:
                return "year_above_max"

        if e.types:
            if not work.type:
                if strict:
                    return "no_type"
            elif work.type not in e.types:
                return "type_not_allowed"

        if e.languages:
            if not work.language:
                if strict:
                    return "no_language"
            elif work.language not in e.languages:
                return "language_not_allowed"

        return None

    def _eligible(self, work: Work, iteration: int) -> bool:
        return self._ineligibility_reason(work) is None

    def _rank(self, candidates: Sequence[Work]) -> dict[str, float]:
        included = self.state.included
        self.ranker.fit(included)
        return dict(self.ranker.score(candidates))

    def _export_candidates(
        self, iteration: int, works: Sequence[Work], scores: Mapping[str, float]
    ) -> None:
        d = self.root / "iterations" / f"iter_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        csv_path = write_candidates_csv(d / "candidates.csv", works, scores)
        write_labels_template(d / "labels_template.csv", works, scores)
        write_ris(d / "candidates.ris", works)
        if self.manifest is not None:
            self.manifest.record_output(
                f"iterations/iter_{iteration:03d}/candidates.csv", csv_path
            )

    def _record_manifest_iteration(
        self,
        iteration: int,
        *,
        batch: Sequence[str],
        n_raw: int,
        n_after_dedup: int,
        n_eligible: int,
    ) -> None:
        if self.manifest is None:
            return
        record = IterationRecord(
            n=iteration,
            frontier_hash=hash_obj(sorted(batch)),
            candidates_hash=hash_obj(list(self.state.pending)),
            n_expanded=len(batch),
            n_raw=n_raw,
            n_after_dedup=n_after_dedup,
            n_eligible=n_eligible,
        )
        self.manifest.record_iteration(record)

    def _finalize_iteration(
        self,
        iteration: int,
        newly_included: Sequence[str],
        scores: Mapping[str, float],
        *,
        n_expanded: int,
        n_raw: int,
        n_screened: int = 0,
        n_included: int = 0,
    ) -> None:
        prev = self.state.history.last
        cumulative_screened = (prev.cumulative_screened if prev else 0) + n_screened
        cumulative_included = len(self.state.included_keys)

        stats = IterationStats(
            n=iteration,
            n_expanded=n_expanded,
            n_raw=n_raw,
            n_after_dedup=n_raw,
            n_eligible=n_screened,
            n_screened=n_screened,
            n_included=n_included,
            n_excluded=max(0, n_screened - n_included),
            cumulative_screened=cumulative_screened,
            cumulative_included=cumulative_included,
            frontier_size=len(self.state.frontier),
        )
        self.state.record_iteration(stats)
        self.state.iteration = iteration

        self.state.history.record_estimate(iteration, self.estimate_recall())
        stop, decisions = self.rules.evaluate(self.state.history)
        self.state.last_decisions = [d.to_dict() for d in decisions]
        for d in decisions:
            self.audit.emit(
                iteration,
                "stop_eval",
                rule=d.rule,
                triggered=d.triggered,
                value=d.value,
                threshold=d.threshold,
            )

        if self.manifest is not None:
            record = next(
                (r for r in self.manifest.iterations if r.n == iteration), None
            )
            if record is None:
                record = IterationRecord(n=iteration, frontier_hash=hash_obj([]))
                self.manifest.record_iteration(record)
            record.n_screened = n_screened
            record.n_included = n_included
            record.n_excluded = stats.n_excluded
            record.stop_decisions = self.state.last_decisions

        if stop:
            self._stop(self.rules.stopped_by(decisions) or "unknown")
        else:
            self.state.phase = Phase.READY
        self._persist()

    def _stop(self, reason: str) -> None:
        self.state.phase = Phase.STOPPED
        self.state.stopped_by = reason
        if self.manifest is not None:
            self.manifest.stopped_by = reason
            self.manifest.finished_at = utcnow().isoformat().replace("+00:00", "Z")
        self.audit.emit(self.state.iteration, "stopped", reason=reason)

    # -- estimation and reporting ---------------------------------------

    def estimate_recall(self, method: str | None = None):
        arms = [a.model_dump() for a in self.config.estimate.arms]
        window = self._cache_window()
        return estimate_recall(
            self.state.included,
            arms,
            method=method or self.config.estimate.method,
            cache_window=window,
            membership=self._membership_sets(),
        )

    def _membership_sets(self):
        """Membership arms, loaded once and cached for the run's lifetime.

        Paths resolve relative to the run directory when not absolute, so a run
        stays portable: a run directory moved alongside its exports still
        estimates. A missing export is a hard error rather than a silently empty
        arm, because an empty arm captures nothing and would make the estimate
        look merely unlucky.
        """
        from ..estimate.membership import MembershipSet

        spec = self.config.estimate.membership_sets
        if not spec:
            return None
        if getattr(self, "_membership_cache", None) is None:
            loaded = {}
            for name, raw in spec.items():
                path = Path(raw)
                if not path.is_absolute() and not path.exists():
                    path = self.root / raw
                if not path.exists():
                    raise ConfigError(
                        f"membership set '{name}' points at a missing export: {raw}"
                    )
                loaded[name] = MembershipSet.from_csv(
                    name, path, doi_column=self.config.estimate.membership_doi_column
                )
            self._membership_cache = loaded
        return self._membership_cache

    def _cache_window(self) -> tuple[str, str] | None:
        stamps = sorted(
            e.retrieved_at for e in self.cache.entries() if e.retrieved_at
        )
        return (stamps[0], stamps[-1]) if stamps else None

    def report(self, *, prisma: bool = True, graph: bool = False) -> dict[str, Path]:
        from ..report.graph import write_graphml
        from ..report.prisma import build_counts, write_prisma_json, write_prisma_svg
        from ..report.summary import write_summary

        out = self.root / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        produced: dict[str, Path] = {}

        if prisma:
            counts = build_counts(self.state, list(self.audit.read()))
            produced["prisma.json"] = write_prisma_json(out / "prisma.json", counts)
            produced["prisma.svg"] = write_prisma_svg(out / "prisma.svg", counts)
        if graph:
            produced["network.graphml"] = write_graphml(out / "network.graphml", self.state)

        produced["included.csv"] = write_candidates_csv(
            out / "included.csv", self.state.included, {}
        )
        produced["report.md"] = write_summary(out / "report.md", self)

        if self.manifest is not None:
            for name, path in sorted(produced.items()):
                self.manifest.record_output(f"outputs/{name}", path)
            self._persist()
        return produced

    def close(self) -> None:
        for p in self.providers:
            p.close()
