"""A third capture-arm design: split the start set, chase each half independently.

Reviewer 2 (revision 1) argues that no workable arm design remains: direction arms
sample disjoint eras because a citation graph is acyclic in time, and provider arms
are not independent captures because OpenAlex ingests a large share of its records
from Crossref. Both objections are properties of *where the records come from*.

This experiment evaluates an arm design that does not depend on the sources at all.
The start set is partitioned at random into two halves; each half is chased
independently to the same depth with the same providers; a relevant record is
"captured" by a chase if that chase discovered it. Overlap is then a property of the
citation graph's connectivity, not of database architecture, so the objection above
does not apply by construction. Whether it yields a usable estimate is the question.
"""
from __future__ import annotations

import json
import random
import statistics as St
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))

import validation_arms as arms
import validation_generator as gen
from snowballslr.estimate.capture_recapture import chapman

PROV = ("p0", "p1", "p2")
COV = {"sparse": (0.55, 0.40, 0.30), "medium": (0.70, 0.55, 0.42), "dense": (0.86, 0.72, 0.60)}
PIN = {"sparse": 0.020, "medium": 0.035, "dense": 0.055}


def relevant_found(res, gold):
    return {k for k in res.found if k in gold}


def one_review(seed, regime, n_seeds, depth, rng):
    spec = gen.GraphSpec(seed=seed, provider_coverage=COV[regime],
                         provider_dependence=0.5, p_in=PIN[regime], refs_mean=22.0)
    g = gen.generate_graph(spec)
    gold = set(g["gold"])
    pool = sorted(gold)
    seeds = rng.sample(pool, min(n_seeds, len(pool)))
    half = len(seeds) // 2
    sA, sB = seeds[:half], seeds[half:]
    rA = arms.expand(g, sA, gold, max_iter=depth, providers=PROV)
    rB = arms.expand(g, sB, gold, max_iter=depth, providers=PROV)
    A, B = relevant_found(rA, gold), relevant_found(rB, gold)
    obs = A | B
    n1, n2, m = len(A), len(B), len(A & B)
    row = {"seed": seed, "regime": regime, "n_seeds": n_seeds, "depth": depth,
           "truth": len(gold), "observed": len(obs), "n1": n1, "n2": n2, "m": m,
           "true_recall": len(obs) / len(gold) if gold else None}
    try:
        e = chapman(n1, n2, m, n_observed=len(obs))
        row["chapman"] = {"estimable": e.estimable, "n_hat": e.n_hat,
                          "n_hat_ci": list(e.n_hat_ci) if e.n_hat_ci else None,
                          "reason": e.reason}
    except Exception as ex:
        row["chapman"] = {"estimable": False, "reason": str(ex)[:80]}
    # provider-arm comparison on the union chase, same graph and same start set
    rU = arms.expand(g, seeds, gold, max_iter=depth, providers=PROV)
    P = {}
    for pj in ("p0", "p1"):
        P[pj] = {k for k, v in rU.found.items() if k in gold and pj in v["provs"]}
    obsU = relevant_found(rU, gold)
    try:
        e2 = chapman(len(P["p0"]), len(P["p1"]), len(P["p0"] & P["p1"]), n_observed=len(obsU))
        row["provider"] = {"estimable": e2.estimable, "n_hat": e2.n_hat,
                           "observed": len(obsU), "m": len(P["p0"] & P["p1"])}
    except Exception as ex:
        row["provider"] = {"estimable": False, "reason": str(ex)[:80]}
    return row


def main():
    quick = "--quick" in sys.argv
    seeds = range(2) if quick else range(12)
    sizes = (10,) if quick else (4, 8, 16, 24)
    depths = (3,) if quick else (1, 2, 3)
    rng = random.Random(7)
    out = []
    for regime in COV:
        for s in seeds:
            for ns in sizes:
                for dp in depths:
                    out.append(one_review(s, regime, ns, dp, rng))
        print(f"  {regime}: {len(out)} rows")
    dest = ROOT / "cases" / ("seedsplit_quick.json" if quick else "seedsplit_study.json")
    dest.write_text(json.dumps({"design": {"providers": list(PROV), "coverage": {k: list(v) for k, v in COV.items()},
                                           "p_in": PIN, "seeds": list(seeds), "sizes": list(sizes),
                                           "depths": list(depths)}, "rows": out}, indent=1))
    est = [r for r in out if r["chapman"].get("estimable")]
    print(f"rows={len(out)} estimable={len(est)} -> {dest.name}")
    if est:
        bias = St.median((r["chapman"]["n_hat"] - r["truth"]) / r["truth"] for r in est)
        le = sum(1 for r in est if r["chapman"]["n_hat"] <= r["truth"])
        print(f"  seed-split: median rel bias {bias:+.3f} | N_hat<=N in {le/len(est):.1%}")


if __name__ == "__main__":
    main()
