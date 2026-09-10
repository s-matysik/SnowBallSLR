"""R2.6 -- coverage study of the three recall estimators on known ground truth.

For every simulated review we expand to exhaustion, and at each iteration compute
Chapman, Chao1 and log-linear over three provider arms. Because the generator knows
the true relevant set, we can score each estimator on bias and on empirical coverage
of its nominal 95% interval -- the quantity a user needs in order to trust a stop
that is triggered by an estimate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))

import validation_generator as gen
import validation_arms as arms_mod
from snowballslr.estimate.capture_recapture import chapman
from snowballslr.estimate.chao import chao1
from snowballslr.estimate.loglinear import loglinear

ARMS = ("p0", "p1", "p2")
# three providers so that all three estimators are computable on identical data
COV = {"sparse": (0.55, 0.40, 0.30), "medium": (0.70, 0.55, 0.42),
       "dense":  (0.86, 0.72, 0.60)}
PIN = {"sparse": 0.020, "medium": 0.035, "dense": 0.055}


def histories(found, subset, arms=ARMS):
    return {k: sorted(a for a in found[k]["provs"] if a in arms)
            for k in subset if k in found and found[k]["provs"]}


def one_review(regime, gseed, n_seeds, strategy):
    spec = gen.GraphSpec(seed=gseed, provider_coverage=COV[regime], p_in=PIN[regime])
    g = gen.generate_graph(spec)
    gold = set(g["gold"]); truth = len(gold)
    pool = sorted(gold)
    yr = {k: g["works"][k]["year"] for k in pool}
    if strategy == "random":
        seeds = pool[:: max(1, len(pool)//n_seeds)][:n_seeds]
    elif strategy == "early":
        seeds = sorted(pool, key=lambda k: yr[k])[:n_seeds]
    else:
        seeds = sorted(pool, key=lambda k: -yr[k])[:n_seeds]

    rows = []
    for depth in range(1, 7):
        ex = arms_mod.expand(g, seeds, gold, max_iter=depth, providers=ARMS)
        inc = ex.included
        if len(inc) < 4:
            continue
        H = histories(ex.found, inc)
        if not H:
            continue
        rec = {"regime": regime, "gseed": gseed, "n_seeds": n_seeds,
               "strategy": strategy, "depth": depth, "truth": truth,
               "observed": len(inc), "true_recall": len(inc)/truth}
        n1, n2, m = arms_mod.arm_counts(ex.found, inc, "provs", "p0", "p1")
        e = chapman(n1, n2, m, n_observed=len(inc))
        rec["chapman"] = e.to_dict()
        try:
            rec["chao1"] = chao1(H).to_dict()
        except Exception as err:
            rec["chao1"] = {"estimable": False, "reason": str(err)[:80]}
        try:
            rec["loglinear"] = loglinear(H, list(ARMS)).to_dict()
        except Exception as err:
            rec["loglinear"] = {"estimable": False, "reason": str(err)[:80]}
        rows.append(rec)
    return rows


def main():
    quick = "--quick" in sys.argv
    gseeds = range(2) if quick else range(10)
    sizes = (8,) if quick else (5, 10, 20)
    strats = ("random",) if quick else ("random", "early", "late")
    out = []
    cells = 0
    for regime in COV:
        for gs in gseeds:
            for ns in sizes:
                for stt in strats:
                    cells += 1
                    out.extend(one_review(regime, gs, ns, stt))
        print(f"  {regime}: {cells} cells, {len(out)} rows", flush=True)
    dest = ROOT / "cases" / ("coverage_quick.json" if quick else "coverage_study.json")
    dest.write_text(json.dumps({"design": {
        "cells": cells, "arms": list(ARMS), "coverage": {k: list(v) for k, v in COV.items()},
        "p_in": PIN, "depths": "1-6", "seed_sizes": list(sizes),
        "strategies": list(strats)}, "rows": out}, indent=1))
    print(f"cells={cells} rows={len(out)} -> {dest.name}")


if __name__ == "__main__":
    main()
