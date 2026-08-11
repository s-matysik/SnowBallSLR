"""Deterministic deduplication cascade (spec 5.3).

No machine learning, no active learning, no thresholds learned from data. The
cascade is a fixed sequence of rules; the merge order is fully determined by
canonical keys, so ``dedup(shuffle(xs)) == dedup(xs)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import JaroWinkler

from ..types import Work, field_conflicts

__all__ = ["DedupResult", "MergeEvent", "blocking_key", "deduplicate"]


@dataclass(frozen=True, slots=True)
class MergeEvent:
    representative: str
    members: tuple[str, ...]
    tier: str
    score: float | None
    conflicts: dict[str, tuple[object, object]]


@dataclass(frozen=True, slots=True)
class DedupResult:
    works: tuple[Work, ...]
    merges: tuple[MergeEvent, ...]
    aliases: dict[str, str]  # absorbed key -> representative key
    suspicious_clusters: tuple[tuple[str, ...], ...]


def blocking_keys(work: Work) -> tuple[str, ...]:
    """Blocking keys for a record; two records are compared if any key matches.

    A single ``year // 2`` bucket silently defeats the +/-1 year tolerance that
    tiers T2 and T3 apply: 1995 and 1996 fall in different buckets and are never
    compared, while 1996 and 1997 share one. On the reference corpus this
    separated 6 of 21 title-identical pairs whose years were within tolerance.
    Emitting a key for the record's own bucket *and* its neighbour makes the
    blocking consistent with the tolerance it is meant to accelerate.

    Records with no year are emitted into a dedicated bucket so that they can
    still meet each other -- previously they were all forced into bucket 0
    alongside genuine year-0 records.
    """
    head = (work.title_norm or "")[:12]
    if work.year is None:
        return (f"{head}|noyear",)
    bucket = work.year // 2
    # Two keys, so a pair straddling a bucket boundary still shares one.
    return (f"{head}|{bucket}", f"{head}|{(work.year + 1) // 2}")


def blocking_key(work: Work) -> str:
    """Primary blocking key. Retained for backwards compatibility."""
    return blocking_keys(work)[0]


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, k: str) -> str:
        root = k
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[k] != root:
            self._parent[k], k = root, self._parent[k]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic: the lexicographically smaller key always wins.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k in self._parent:
            out.setdefault(self.find(k), []).append(k)
        return {r: sorted(m) for r, m in out.items()}


def _same_first_author(a: Work, b: Work) -> bool:
    """Compare first-author surnames, tolerating leading initials.

    Providers disagree on how the author field is populated: Crossref
    unstructured references frequently yield ``"AM Cook"`` or ``"G.Amudha"``
    where OpenAlex yields ``"Cook"`` / ``"Amudha"``. After folding to bare
    letters those compare unequal, so genuine duplicates survive every tier.
    Matching the shorter folded surname as a suffix of the longer recovers these
    pairs, with a length floor so that short strings cannot collide by accident.
    """
    from .normalize import normalize_surname

    sa = normalize_surname(a.first_author_surname)
    sb = normalize_surname(b.first_author_surname)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    short, long = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    # The prefix dropped from the longer form must look like initials: at most
    # three characters, i.e. "amcook" vs "cook", never "berg" vs "vandenberg".
    return len(short) >= 4 and long.endswith(short) and len(long) - len(short) <= 3


def _year_close(a: Work, b: Work, tol: int = 1) -> bool:
    if a.year is None or b.year is None:
        return False
    return abs(a.year - b.year) <= tol


def _match_tier(a: Work, b: Work, jw_threshold: float) -> tuple[str, float | None] | None:
    """First matching tier wins. Returns ``(tier, score)`` or ``None``."""
    # T1: identical normalized DOI.
    if a.doi and b.doi and a.doi == b.doi:
        return "T1", None
    # Distinct DOIs are strong evidence of distinct works; block further tiers.
    if a.doi and b.doi and a.doi != b.doi:
        return None
    if not a.title_norm or not b.title_norm:
        return None
    # T2: identical title, year within tolerance, same first author.
    if a.title_norm == b.title_norm and _year_close(a, b) and _same_first_author(a, b):
        return "T2", 1.0
    # T3: fuzzy title, year within tolerance, same first author.
    if _year_close(a, b) and _same_first_author(a, b):
        score = JaroWinkler.similarity(a.title_norm, b.title_norm)
        if score >= jw_threshold:
            return "T3", score
    # T4: identical title and year, author missing on at least one side.
    if (
        a.title_norm == b.title_norm
        and a.year is not None
        and a.year == b.year
        and not (a.first_author_surname and b.first_author_surname)
    ):
        return "T4", 1.0
    return None


def deduplicate(
    works: Sequence[Work],
    *,
    jw_threshold: float = 0.95,
    max_cluster_size: int = 8,
    precedence: Sequence[str] = (),
) -> DedupResult:
    """Cluster and merge duplicate records.

    Clusters larger than ``max_cluster_size`` are aborted (members kept apart)
    and reported as suspicious -- this is the guard against runaway merges on
    generic titles such as "Editorial" or "Introduction".
    """
    ordered = sorted(works, key=lambda w: w.key)
    if not ordered:
        return DedupResult((), (), {}, ())

    by_key: dict[str, Work] = {}
    for w in ordered:
        if w.key in by_key:
            by_key[w.key] = by_key[w.key].merged_with(w, precedence)
        else:
            by_key[w.key] = w
    ordered = [by_key[k] for k in sorted(by_key)]

    uf = _UnionFind(w.key for w in ordered)
    tiers: dict[tuple[str, str], tuple[str, float | None]] = {}

    # Tier 1 runs globally on DOI; tiers 2-4 run inside blocks.
    by_doi: dict[str, list[Work]] = {}
    for w in ordered:
        if w.doi:
            by_doi.setdefault(w.doi, []).append(w)
    for _doi, group in sorted(by_doi.items()):
        for i in range(len(group) - 1):
            a, b = group[i], group[i + 1]
            uf.union(a.key, b.key)
            tiers[tuple(sorted((a.key, b.key)))] = ("T1", None)

    blocks: dict[str, list[Work]] = {}
    for w in ordered:
        for bk in blocking_keys(w):
            blocks.setdefault(bk, []).append(w)

    for _bk, group in sorted(blocks.items()):
        group = sorted(group, key=lambda w: w.key)
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pair = tuple(sorted((a.key, b.key)))
                if pair in tiers:
                    continue
                hit = _match_tier(a, b, jw_threshold)
                if hit is not None:
                    uf.union(a.key, b.key)
                    tiers[pair] = hit

    groups = uf.groups()
    merged: list[Work] = []
    merges: list[MergeEvent] = []
    aliases: dict[str, str] = {}
    suspicious: list[tuple[str, ...]] = []

    for representative in sorted(groups):
        members = groups[representative]
        if len(members) > max_cluster_size:
            suspicious.append(tuple(members))
            merged.extend(by_key[k] for k in members)
            continue
        if len(members) == 1:
            merged.append(by_key[members[0]])
            continue

        acc = by_key[members[0]]
        conflicts: dict[str, tuple[object, object]] = {}
        for k in members[1:]:
            other = by_key[k]
            conflicts.update(field_conflicts(acc, other))
            acc = acc.merged_with(other, precedence)
            aliases[k] = representative
        acc = Work(**{**acc.to_dict(), "key": representative}) if False else acc
        if acc.key != representative:
            acc = _rekey(acc, representative)
        merged.append(acc)

        tier_hits = [tiers[p] for p in tiers if p[0] in members and p[1] in members]
        tier = min((t for t, _ in tier_hits), default="T?")
        score = next((s for t, s in tier_hits if t == tier), None)
        merges.append(
            MergeEvent(
                representative=representative,
                members=tuple(members),
                tier=tier,
                score=score,
                conflicts=conflicts,
            )
        )

    merged.sort(key=lambda w: w.key)
    return DedupResult(
        works=tuple(merged),
        merges=tuple(merges),
        aliases=dict(sorted(aliases.items())),
        suspicious_clusters=tuple(suspicious),
    )


def _rekey(work: Work, key: str) -> Work:
    from dataclasses import replace

    return replace(work, key=key)
