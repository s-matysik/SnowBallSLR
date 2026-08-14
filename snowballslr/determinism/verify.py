"""Verification (spec 9.3).

Two modes:

* **replay** -- recompute every artifact from the cache and compare hashes
  against the manifest. Any mismatch is a bug; this is what enforces INV-1.
* **refresh** -- refetch every cached request against the live API and classify
  what changed. This is the answer to a question no other citation-searching
  tool asks: what does it *mean* to repeat a search against a database that
  keeps moving?

A refresh never mutates the original cache. It writes to ``cache_refresh/``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..errors import VerificationError
from .cache import Cache, CacheEntry
from .manifest import RunManifest, hash_file

__all__ = ["DiffClass", "EntryDiff", "VerifyReport", "verify_refresh", "verify_replay"]


class DiffClass(StrEnum):
    UNCHANGED = "unchanged"
    NEW_CITING = "new_citing"
    LOST_CITING = "lost_citing"
    METADATA_CHANGED = "metadata_changed"
    RETRACTED = "retracted"
    DEINDEXED = "deindexed"
    MERGED = "merged"


@dataclass(frozen=True, slots=True)
class EntryDiff:
    cache_key: str
    provider: str
    method: str
    diff_class: DiffClass
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "provider": self.provider,
            "method": self.method,
            "class": str(self.diff_class),
            "added": list(self.added),
            "removed": list(self.removed),
            "changed_fields": list(self.changed_fields),
            "note": self.note,
        }


@dataclass
class VerifyReport:
    mode: str
    ok: bool
    original_run_started: str | None = None
    checked_at: str | None = None
    artifacts_checked: int = 0
    artifacts_mismatched: list[str] = field(default_factory=list)
    entries_checked: int = 0
    cache_entries_mismatched: list[str] = field(default_factory=list)
    config_drift: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    per_provider: dict[str, dict[str, int]] = field(default_factory=dict)
    diffs: list[EntryDiff] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "original_run_started": self.original_run_started,
            "checked_at": self.checked_at,
            "artifacts_checked": self.artifacts_checked,
            "artifacts_mismatched": sorted(self.artifacts_mismatched),
            "entries_checked": self.entries_checked,
            "cache_entries_mismatched": sorted(self.cache_entries_mismatched),
            "config_drift": self.config_drift,
            "counts": dict(sorted(self.counts.items())),
            "per_provider": {
                k: dict(sorted(v.items())) for k, v in sorted(self.per_provider.items())
            },
            "diffs": [d.to_dict() for d in self.diffs],
        }

    def write(self, run_dir: str | Path) -> tuple[Path, Path]:
        d = Path(run_dir) / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        json_path = d / "verify_report.json"
        json_path.write_text(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path = d / "verify_report.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self) -> str:
        lines = [f"# Verification report ({self.mode})", ""]
        lines.append(f"- Result: {'PASS' if self.ok else 'FAIL'}")
        if self.original_run_started:
            lines.append(f"- Original run started: {self.original_run_started}")
        if self.checked_at:
            lines.append(f"- Verified at: {self.checked_at}")
        lines.append(f"- Artifacts checked: {self.artifacts_checked}")
        if self.artifacts_mismatched:
            lines.append(f"- Artifacts mismatched: {len(self.artifacts_mismatched)}")
        lines.append(f"- Cache entries checked: {self.entries_checked}")
        lines.append("")

        if self.mode == "replay":
            if self.artifacts_mismatched:
                lines.append("## Mismatched artifacts")
                lines.append("")
                for name in sorted(self.artifacts_mismatched):
                    lines.append(f"- `{name}`")
                lines.append("")
                lines.append(
                    "A replay mismatch means the run is not reproducible from its own "
                    "cache. This is a defect, not database drift."
                )
            else:
                lines.append(
                    "Every recorded artifact reproduced byte-for-byte from the cache."
                )
            lines.append("")
            return "\n".join(lines)

        lines.append("## Drift classification")
        lines.append("")
        lines.append("| Class | Entries |")
        lines.append("|---|---:|")
        for name in [c.value for c in DiffClass]:
            lines.append(f"| `{name}` | {self.counts.get(name, 0)} |")
        lines.append("")

        if self.per_provider:
            lines.append("## By provider")
            lines.append("")
            lines.append("| Provider | " + " | ".join(c.value for c in DiffClass) + " |")
            lines.append("|---" * (len(DiffClass) + 1) + "|")
            for provider, counts in sorted(self.per_provider.items()):
                row = " | ".join(str(counts.get(c.value, 0)) for c in DiffClass)
                lines.append(f"| {provider} | {row} |")
            lines.append("")

        notable = [
            d
            for d in self.diffs
            if d.diff_class
            in (DiffClass.RETRACTED, DiffClass.DEINDEXED, DiffClass.MERGED)
        ]
        if notable:
            lines.append("## Records requiring attention")
            lines.append("")
            for d in notable[:100]:
                lines.append(f"- `{d.diff_class}` -- {d.provider}.{d.method}: {d.note}")
            lines.append("")

        lines.append(
            "_Drift is expected: OpenAlex and Semantic Scholar are living databases. "
            "What matters for reproducibility is that the drift is measured and "
            "classified rather than silently absorbed into a different result._"
        )
        lines.append("")
        return "\n".join(lines)


# -- replay ---------------------------------------------------------------


def _verify_cache_integrity(cache: Cache) -> list[str]:
    """Re-check every cache entry against its own two content addresses.

    An entry is stored at a path derived from ``provider|method|params`` and
    carries ``response_hash`` over its body. Recomputing both detects a tampered
    or truncated cache, which the artifact-hash pass alone cannot see.
    """
    from ..config import canonical_json
    from .cache import cache_key as _cache_key
    from .cache import sha256_of

    bad: list[str] = []
    # `Cache.keys()` is a method returning a sorted list of on-disk entry names,
    # not a mapping view.
    for key in cache.keys():  # noqa: SIM118
        try:
            entry = cache.read(key)
        except Exception:
            bad.append(f"{key} (unreadable)")
            continue
        if entry is None:
            bad.append(f"{key} (missing)")
            continue
        expected_key = _cache_key(entry.provider, entry.method, entry.params)
        if expected_key != entry.cache_key or expected_key != key:
            bad.append(f"{key} (request key does not match its contents)")
            continue
        if sha256_of(canonical_json(entry.body)) != entry.response_hash:
            bad.append(f"{key} (body does not match its recorded hash)")
    return sorted(bad)


def _only_new_schema_keys(stored: Mapping[str, Any], config_cls: Any) -> bool:
    """True when reloading the snapshot adds settings but changes none it recorded.

    A run written before a setting existed stores no key for it, so reloading supplies
    that setting's default and the hash moves. That is the schema advancing, not the
    configuration being altered, and failing on it would make `verify` useless against
    any run older than the current release.

    This must be narrow. Rehashing an EDITED snapshot reproduces the edited value, so
    the round-trip agrees with itself and only the mismatch against the manifest's
    recorded hash reveals tampering. The allowance therefore applies only when the
    round-trip genuinely ADDS keys and leaves every recorded value untouched.
    """
    try:
        reloaded = config_cls.from_dict(stored).hashable()
    except Exception:
        return False

    added = False

    def walk(a: Any, b: Any) -> bool:
        nonlocal added
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in set(a) | set(b):
                if key not in a:
                    added = True  # a setting the run predates
                    continue
                if key not in b:
                    return False  # the run recorded a value the schema dropped
                if a[key] is None and b[key] is not None:
                    added = True  # stored as null, now carries a default
                    continue
                if not walk(a[key], b[key]):
                    return False
            return True
        return bool(a == b)

    return walk(stored, reloaded) and added


def _changed_settings(
    stored: Mapping[str, Any], live: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Settings the run recorded whose value has since changed, as (path, was, now).

    Keys the run did not record are skipped: their value in `live` is whatever the
    current schema defaults to, which says nothing about the run.
    """
    out: list[tuple[str, Any, Any]] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Mapping):
            if not isinstance(b, Mapping):
                out.append((path, a, b))
                return
            for key in sorted(a):
                if a[key] is None:
                    continue
                walk(a[key], b.get(key), f"{path}.{key}" if path else str(key))
        elif a != b:
            out.append((path, a, b))

    walk(stored, live, "")
    return out


def _verify_config_hash(root: Path, manifest: RunManifest) -> str | None:
    """Return a description of any drift between the live config and the manifest.

    Two distinct things can go wrong, and both must be checked:

    1. The manifest's own snapshot no longer hashes to its recorded hash -- the
       snapshot was edited apart from the hash, or the schema has moved under it.
    2. `<run_dir>/config.yaml`, which `Run.load` reads and which therefore drives
       every regenerated report, no longer matches the configuration the run
       executed under. Raising a stopping threshold or an iteration cap to continue
       a stopped run writes this file, so the run directory ends up holding a report
       whose config hash the manifest cannot corroborate.

    Checking only (1) is not enough: it compares the snapshot against itself and
    passes on exactly the drift that motivated this check.
    """
    from ..config import Config

    recorded = manifest.config_hash
    try:
        snapshot = Config.from_dict(manifest.config).config_hash
    except Exception as exc:  # a snapshot the current schema cannot load is itself drift
        return f"manifest config could not be reloaded under the current schema: {exc}"
    if snapshot != recorded and not _only_new_schema_keys(manifest.config, Config):
        # A snapshot that rehashes differently is drift ONLY if the difference is in
        # values the run actually set. A run predating a later-added setting stores no
        # key for it, so reloading supplies that setting's default and moves the hash --
        # schema evolution, not tampering, and failing on it would make `verify` useless
        # against any run older than the current release.
        return (
            f"the manifest's config snapshot hashes to {snapshot} but the manifest "
            f"records {recorded}; the snapshot was altered apart from its hash, so the "
            f"manifest no longer describes the run"
        )

    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        live_cfg = Config.from_yaml(cfg_path)
    except Exception as exc:
        return f"{cfg_path.name} could not be loaded under the current schema: {exc}"
    live = live_cfg.config_hash
    if live == recorded:
        return None
    # Both sides are hashed through the current schema here, so a bare hash difference
    # can still be schema evolution: the snapshot predates a setting whose default the
    # live file now carries explicitly. Compare the values the run actually recorded.
    changed = _changed_settings(manifest.config, live_cfg.hashable())
    if not changed:
        return None
    detail = "; ".join(f"{path}: {was!r} -> {now!r}" for path, was, now in changed[:4])
    return (
        f"{cfg_path.name} hashes to {live} but the run executed under {recorded} "
        f"({detail}); the configuration was changed after the run was written, so any "
        f"report regenerated from it describes a configuration this run did not use"
    )


def verify_replay(run_dir: str | Path, *, strict: bool = True) -> VerifyReport:
    """Recompute artifacts from cache and compare against the manifest."""
    root = Path(run_dir)
    manifest_path = root / "run.json"
    if not manifest_path.exists():
        raise VerificationError(f"no manifest at {manifest_path}")
    manifest = RunManifest.load(manifest_path)

    from ..types import utcnow

    report = VerifyReport(
        mode="replay",
        ok=True,
        original_run_started=manifest.started_at,
        checked_at=utcnow().isoformat().replace("+00:00", "Z"),
    )

    for name, expected in sorted(manifest.outputs.items()):
        path = root / name
        report.artifacts_checked += 1
        if not path.exists():
            report.artifacts_mismatched.append(f"{name} (missing)")
            continue
        if hash_file(path) != expected:
            report.artifacts_mismatched.append(name)

    # The manifest records the configuration the run actually executed under. If the
    # configuration on disk has since changed -- a stopping threshold edited, an
    # iteration cap raised to let a stopped run continue -- then the manifest no
    # longer describes the run, and any report regenerated from the live config will
    # quote a different config hash than the manifest it claims to match. Comparing
    # the two is the only way that drift becomes visible.
    report.config_drift = _verify_config_hash(root, manifest)

    cache = Cache(root / manifest.config.get("determinism", {}).get("cache_dir", "cache"))
    report.entries_checked = len(cache)

    # The cache is the substrate a replay would be recomputed from, so checking
    # only the emitted artifacts leaves the claim half-verified: a corrupted or
    # hand-edited cache body previously passed `verify` untouched. Every entry is
    # content-addressed twice over -- its filename is the request key and it
    # carries the hash of its own body -- so both are cheap to re-check.
    report.cache_entries_mismatched.extend(_verify_cache_integrity(cache))

    report.ok = (
        not report.artifacts_mismatched
        and not report.cache_entries_mismatched
        and report.config_drift is None
    )

    if strict and not report.ok:
        report.write(root)
        raise VerificationError(
            f"replay mismatch in {len(report.artifacts_mismatched)} artifact(s): "
            f"{sorted(report.artifacts_mismatched)[:5]}"
        )
    return report


# -- refresh --------------------------------------------------------------

_ID_FIELDS = ("id", "DOI", "doi", "paperId", "corpusId")
_TRACKED_FIELDS = (
    "title",
    "display_name",
    "publication_year",
    "year",
    "type",
    "language",
    "is_retracted",
    "venue",
)


def _collect_ids(body: Any) -> set[str]:
    """Identifier set of a response body, used to detect membership changes."""
    out: set[str] = set()
    if isinstance(body, Mapping):
        for field_name in ("results", "data", "items"):
            rows = body.get(field_name)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    out |= _collect_ids(row)
        for nested in ("citedPaper", "citingPaper", "message"):
            if nested in body:
                out |= _collect_ids(body[nested])
        for key in _ID_FIELDS:
            value = body.get(key)
            if isinstance(value, (str, int)):
                out.add(f"{key}:{value}")
        refs = body.get("referenced_works")
        if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
            out |= {f"ref:{r}" for r in refs if isinstance(r, str)}
    elif isinstance(body, Sequence) and not isinstance(body, (str, bytes)):
        for row in body:
            out |= _collect_ids(row)
    return out


def _primary_record(body: Any) -> Mapping[str, Any] | None:
    if isinstance(body, Mapping):
        if "message" in body and isinstance(body["message"], Mapping):
            return body["message"]
        if any(k in body for k in _ID_FIELDS):
            return body
    return None


def classify(old: CacheEntry, new_body: Any, new_status: int) -> EntryDiff:
    """Classify the difference between a cached response and a fresh one."""
    base = {
        "cache_key": old.cache_key,
        "provider": old.provider,
        "method": old.method,
    }

    if new_status == 404 or new_body is None:
        return EntryDiff(
            **base,
            diff_class=DiffClass.DEINDEXED,
            note=f"{old.method} target no longer resolvable ({new_status})",
        )

    from ..config import canonical_json
    from .cache import sha256_of

    if sha256_of(canonical_json(new_body)) == old.response_hash:
        return EntryDiff(**base, diff_class=DiffClass.UNCHANGED)

    old_rec = _primary_record(old.body)
    new_rec = _primary_record(new_body)

    if old_rec is not None and new_rec is not None:
        if not old_rec.get("is_retracted") and new_rec.get("is_retracted"):
            return EntryDiff(
                **base,
                diff_class=DiffClass.RETRACTED,
                note=f"record flagged retracted: {new_rec.get('id') or new_rec.get('DOI')}",
            )
        old_id = str(old_rec.get("id") or "")
        new_id = str(new_rec.get("id") or "")
        if old_id and new_id and old_id != new_id:
            return EntryDiff(
                **base,
                diff_class=DiffClass.MERGED,
                note=f"identifier redirected: {old_id} -> {new_id}",
            )

    old_ids = _collect_ids(old.body)
    new_ids = _collect_ids(new_body)
    added = tuple(sorted(new_ids - old_ids))
    removed = tuple(sorted(old_ids - new_ids))

    if added and not removed:
        return EntryDiff(**base, diff_class=DiffClass.NEW_CITING, added=added)
    if removed and not added:
        return EntryDiff(**base, diff_class=DiffClass.LOST_CITING, removed=removed)
    if added or removed:
        return EntryDiff(
            **base, diff_class=DiffClass.NEW_CITING, added=added, removed=removed
        )

    changed: list[str] = []
    if old_rec is not None and new_rec is not None:
        for f in _TRACKED_FIELDS:
            if (f in old_rec or f in new_rec) and old_rec.get(f) != new_rec.get(f):
                changed.append(f)
    return EntryDiff(
        **base,
        diff_class=DiffClass.METADATA_CHANGED,
        changed_fields=tuple(sorted(changed)),
        note="response differs without membership change",
    )


def verify_refresh(
    run_dir: str | Path,
    fetcher: Any,
    *,
    limit: int | None = None,
) -> VerifyReport:
    """Refetch every cached request and classify the drift.

    ``fetcher`` is a callable ``(entry) -> (body, status)``. Injecting it keeps
    this function testable offline (INV-4) and lets the CLI supply live
    providers.
    """
    root = Path(run_dir)
    manifest = RunManifest.load(root / "run.json")
    cache_dir = manifest.config.get("determinism", {}).get("cache_dir", "cache")
    original = Cache(root / cache_dir)
    refreshed = Cache(root / "cache_refresh")

    from ..types import utcnow

    report = VerifyReport(
        mode="refresh",
        ok=True,
        original_run_started=manifest.started_at,
        checked_at=utcnow().isoformat().replace("+00:00", "Z"),
    )

    entries = list(original.entries())
    if limit is not None:
        entries = entries[:limit]

    for entry in entries:
        report.entries_checked += 1
        body, status = fetcher(entry)
        refreshed.write(
            CacheEntry(
                cache_key=entry.cache_key,
                provider=entry.provider,
                method=entry.method,
                params=entry.params,
                retrieved_at=report.checked_at or "",
                http_status=status,
                response_hash=entry.response_hash,
                provider_version=entry.provider_version,
                body=body,
            )
        )
        diff = classify(entry, body, status)
        report.diffs.append(diff)
        name = str(diff.diff_class)
        report.counts[name] = report.counts.get(name, 0) + 1
        report.per_provider.setdefault(entry.provider, {})
        report.per_provider[entry.provider][name] = (
            report.per_provider[entry.provider].get(name, 0) + 1
        )

    report.ok = report.counts.get(DiffClass.DEINDEXED.value, 0) == 0
    return report
