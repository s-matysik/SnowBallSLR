"""Deterministic synthetic citation-graph generator for validation experiments.

Generates a temporally-ordered citation DAG with a known relevant population, so
stopping rules and recall estimators can be scored against ground truth.

Design choices that matter for external validity:

* **Temporal acyclicity.** A work may only cite works published no later than
  itself. This is not a convenience: it is the reason backward and forward
  expansion sample near-disjoint strata of the graph, which is what breaks
  direction-based capture-recapture arms.
* **Topic homophily.** Relevant works cite relevant works with probability
  ``p_in``, irrelevant works with ``p_out < p_in``. The ratio controls how
  navigable the relevant subgraph is by citation chasing alone.
* **Partial reference coverage.** Each edge is independently visible to each
  provider with a per-provider probability, and a shared latent "indexability"
  per work induces *positive dependence* between providers -- the condition under
  which capture-recapture is biased.
* **Disconnected relevant works.** A tunable share of relevant works has no
  citation path to the rest, imposing a hard ceiling on achievable recall that no
  stopping rule can cross. Real reviews have this ceiling; a benchmark without it
  flatters every rule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

__all__ = ["GraphSpec", "generate_graph"]


@dataclass(frozen=True, slots=True)
class GraphSpec:
    n_relevant: int = 120
    n_irrelevant: int = 900
    year_min: int = 2005
    year_max: int = 2024
    refs_mean: float = 28.0
    p_in: float = 0.055           # relevant -> relevant citation probability
    p_out: float = 0.004          # relevant -> irrelevant / irrelevant -> any
    disconnected_share: float = 0.08
    provider_coverage: tuple[float, ...] = (0.86, 0.72)
    provider_dependence: float = 0.6   # 0 = independent, 1 = fully shared latent
    unresolved_share: float = 0.10
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_relevant": self.n_relevant, "n_irrelevant": self.n_irrelevant,
            "year_min": self.year_min, "year_max": self.year_max,
            "refs_mean": self.refs_mean, "p_in": self.p_in, "p_out": self.p_out,
            "disconnected_share": self.disconnected_share,
            "provider_coverage": list(self.provider_coverage),
            "provider_dependence": self.provider_dependence,
            "unresolved_share": self.unresolved_share, "seed": self.seed,
        }


def generate_graph(spec: GraphSpec) -> dict[str, Any]:
    """Return {works, references, citations, gold, provider_edges, spec}."""
    rng = random.Random(spec.seed)
    n_rel, n_irr = spec.n_relevant, spec.n_irrelevant
    total = n_rel + n_irr

    keys, meta = [], {}
    for i in range(total):
        relevant = i < n_rel
        key = f"doi:10.{'1000' if relevant else '2000'}/{'rel' if relevant else 'irr'}{i:05d}"
        # Recent-weighted year distribution, as in real corpora.
        span = spec.year_max - spec.year_min
        year = spec.year_min + int(span * (rng.random() ** 0.65))
        keys.append(key)
        meta[key] = {"relevant": relevant, "year": year, "idx": i}

    # A share of relevant works is deliberately unreachable by citation chasing.
    rel_keys = [k for k in keys if meta[k]["relevant"]]
    n_disc = int(round(spec.disconnected_share * len(rel_keys)))
    disconnected = set(rng.sample(rel_keys, n_disc)) if n_disc else set()

    # Latent indexability drives provider dependence.
    latent = {k: rng.random() for k in keys}

    references: dict[str, list[str]] = {k: [] for k in keys}
    provider_edges: dict[str, dict[str, list[str]]] = {
        f"p{j}": {k: [] for k in keys} for j in range(len(spec.provider_coverage))
    }

    # Whether a provider indexes a record is decided ONCE per (provider, record). The
    # shared latent term makes the two providers positively dependent -- a well-indexed
    # work tends to be carried by both -- while the independent term leaves room for
    # provider-only capture, which is the signal capture-recapture arms rely on.
    indexed: dict[str, dict[str, bool]] = {}
    for j, cov in enumerate(spec.provider_coverage):
        pj = f"p{j}"
        indexed[pj] = {}
        for k in keys:
            mix = (
                spec.provider_dependence * latent[k]
                + (1 - spec.provider_dependence) * rng.random()
            )
            indexed[pj][k] = mix < cov

    by_year: dict[int, list[str]] = {}
    for k in keys:
        by_year.setdefault(meta[k]["year"], []).append(k)

    for src in keys:
        if src in disconnected:
            continue
        y_src = meta[src]["year"]
        src_rel = meta[src]["relevant"]
        older = [k for k in keys if meta[k]["year"] <= y_src and k != src and k not in disconnected]
        if not older:
            continue
        # A reference list has a roughly fixed budget; homophily governs its
        # COMPOSITION, not its size. Sampling weights therefore decide the mix
        # while n_refs decides the count -- otherwise the two are entangled and
        # a low p_out silently produces two-reference papers.
        n_refs = max(0, int(rng.gauss(spec.refs_mean, spec.refs_mean * 0.35)))
        n_refs = min(n_refs, len(older))
        pool = older if len(older) <= 600 else rng.sample(older, 600)
        weights = []
        for tgt in pool:
            w = spec.p_in if (src_rel and meta[tgt]["relevant"]) else spec.p_out
            age = y_src - meta[tgt]["year"]
            w *= 1.0 + 0.04 * min(age, 12)          # preferential attachment to older work
            w *= 1.0 + 2.0 * latent[tgt]            # citation-count heterogeneity
            weights.append(w)
        n_take = min(n_refs, len(pool))
        chosen: set[str] = set()
        remaining = list(zip(pool, weights))
        for _ in range(n_take):
            total_w = sum(w for _, w in remaining)
            if total_w <= 0:
                break
            r = rng.random() * total_w
            acc = 0.0
            for i, (cand, w) in enumerate(remaining):
                acc += w
                if acc >= r:
                    chosen.add(cand)
                    remaining.pop(i)
                    break
        references[src] = sorted(chosen)

        for j, cov in enumerate(spec.provider_coverage):
            pj = f"p{j}"
            for tgt in references[src]:
                # Indexing is a property of the RECORD, not of each edge pointing at
                # it. Drawing per edge made a work's chance of being missed 0.28**d in
                # its in-degree d, so at a median in-degree of ~21 every work was
                # visible to every provider and simulated arm overlap was 100% -- while
                # the live runs show 6% to 45%. Deciding once per (provider, record)
                # reproduces provider-only capture, which is what the arms measure.
                if indexed[pj][tgt]:
                    provider_edges[pj][src].append(tgt)

    citations: dict[str, list[str]] = {k: [] for k in keys}
    for src, tgts in references.items():
        for t in tgts:
            citations[t].append(src)
    citations = {k: sorted(v) for k, v in citations.items()}

    works = {}
    n_unres = int(round(spec.unresolved_share * total))
    unresolved = set(rng.sample(keys, n_unres)) if n_unres else set()
    for k in keys:
        m = meta[k]
        works[k] = {
            "key": k,
            "title": ("Relevant" if m["relevant"] else "Unrelated")
                     + f" study {m['idx']} on citation searching"
                     + ("" if m["relevant"] else " in an adjacent field"),
            "title_norm": "",
            "doi": k.split(":", 1)[1],
            "abstract": ("snowballing citation searching systematic review saturation"
                         if m["relevant"] else "unrelated topic adjacent field"),
            "year": m["year"],
            "type": "article",
            "language": "en",
            "is_retracted": False,
            "unresolved": k in unresolved,
            "venue": "Journal of Synthesis",
            "reference_count": len(references[k]),
            "cited_by_count": len(citations[k]),
            "source_ids": {"openalex": f"W{m['idx']:07d}"},
            "provenance": [],
        }

    return {
        "works": works,
        "references": {k: v for k, v in references.items() if v},
        "citations": {k: v for k, v in citations.items() if v},
        "gold": sorted(rel_keys),
        "disconnected": sorted(disconnected),
        "provider_edges": provider_edges,
        "spec": spec.to_dict(),
    }
