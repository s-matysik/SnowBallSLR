"""E1 -- arm design for capture-recapture on a citation graph.

Simulates snowballing expansion while tracking, for every discovered work, which
"arm" discovered it under three candidate designs: direction arms
(backward/forward), provider arms (p0/p1), and a mixed design. Runs against the
synthetic generator so the true relevant population is known.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ExpansionResult:
    found: dict[str, dict[str, set]]   # key -> {"dirs": set, "provs": set}
    included: set
    iterations: int
    screened: int


def expand(graph: dict, seeds: list[str], gold: set, *, max_iter: int = 6,
           directions=("backward", "forward"), providers=("p0", "p1")) -> ExpansionResult:
    """Mirror the library's frontier semantics, tracking arm membership."""
    refs, cits = graph["references"], graph["citations"]
    pedges = graph["provider_edges"]

    # provider-visible reverse index for forward direction
    fwd_by_prov = {}
    for pj in providers:
        rev: dict[str, list[str]] = {}
        for src, tgts in pedges[pj].items():
            for t in tgts:
                rev.setdefault(t, []).append(src)
        fwd_by_prov[pj] = rev

    found: dict[str, dict[str, set]] = {}
    decided: set = set(seeds)
    included: set = set(s for s in seeds if s in gold)
    frontier = list(seeds)
    screened = 0
    iterations = 0

    for _ in range(max_iter):
        if not frontier:
            break
        iterations += 1
        batch, frontier = frontier, []
        newly: set = set()
        for parent in batch:
            for pj in providers:
                if "backward" in directions:
                    for t in pedges[pj].get(parent, []):
                        rec = found.setdefault(t, {"dirs": set(), "provs": set()})
                        rec["dirs"].add("backward"); rec["provs"].add(pj)
                        if t not in decided:
                            newly.add(t)
                if "forward" in directions:
                    for s in fwd_by_prov[pj].get(parent, []):
                        rec = found.setdefault(s, {"dirs": set(), "provs": set()})
                        rec["dirs"].add("forward"); rec["provs"].add(pj)
                        if s not in decided:
                            newly.add(s)
        if not newly:
            break
        screened += len(newly)
        for k in sorted(newly):
            decided.add(k)
            if k in gold:
                included.add(k)
                frontier.append(k)
    return ExpansionResult(found=found, included=included,
                           iterations=iterations, screened=screened)


def arm_counts(found: dict, subset: set, field: str, a: str, b: str) -> tuple[int, int, int]:
    n1 = sum(1 for k in subset if k in found and a in found[k][field])
    n2 = sum(1 for k in subset if k in found and b in found[k][field])
    m  = sum(1 for k in subset if k in found and a in found[k][field] and b in found[k][field])
    return n1, n2, m
