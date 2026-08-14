"""E6 -- does the recall rule still fire when arm overlap matches live data?

The benchmark in `validation_stopping.py` reports the recall rule firing in 127 of
270 simulated reviews, while none of the four live runs reached a rule-driven stop
on that rule. This experiment tests whether the difference is a property of the
GENERATOR rather than of the rule.

Two generator settings were measured against the live runs and found unrealistic:

* Provider coverage of (0.86, 0.72) with dependence 0.6 makes both arms capture
  almost every record, so simulated arm overlap is ~99%. The four live runs show
  6%, 12%, 24% and 45%.
* `p_in = 0.055` over 1020 works gives relevant records a median in-degree of ~21,
  so each is reached by many parents and accumulates both arms. Live medians are
  1 to 2 distinct parents, with 32% to 98% of included records reached exactly once.

Chapman's estimator needs overlap to identify the population. With overlap near 1
the estimate collapses onto the observed count, recall reads high, and the rule
fires. At live overlap the estimate is wide and the lower bound stays below tau --
which is exactly what the live runs show. This experiment quantifies that.

Usage: python validation/validation_stopping_calibrated.py [--quick]
"""

from __future__ import annotations

import argparse
import random
import statistics as St
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import validation_arms as arms  # noqa: E402
import validation_generator as gen  # noqa: E402

from snowballslr.core.iteration import IterationHistory, IterationStats  # noqa: E402
from snowballslr.estimate.capture_recapture import chapman  # noqa: E402
from snowballslr.stopping.recall_rule import EstimatedRecall  # noqa: E402

# Measured on the four live runs: (provider-only, provider-only, both) over included
# records. Overlap = both / total.
LIVE_OVERLAP = {"run_large": 0.061, "run_llm": 0.123, "run_mid": 0.236,
                "run_adverse": 0.433, "run_narrow": 0.447}

# The generator's shipped defaults, and a setting whose arm overlap falls inside the
# live band. Only the provider model and graph density differ.
OPTIMISTIC = dict(provider_coverage=(0.86, 0.72), provider_dependence=0.6,
                  p_in=0.055, refs_mean=28.0)
CALIBRATED = dict(provider_coverage=(0.45, 0.22), provider_dependence=0.4,
                  p_in=0.010, refs_mean=22.0)


def _one_review(spec_kw: dict, seed: int, n_seeds: int, max_iter: int) -> dict:
    """Expand to exhaustion, evaluating the guarded rule at every iteration."""
    graph = gen.generate_graph(gen.GraphSpec(seed=seed, **spec_kw))
    gold = set(graph["gold"])
    seeds = random.Random(1000 + seed).sample(sorted(gold), n_seeds)

    history = IterationHistory()
    rule = EstimatedRecall(tau=0.95, use_lower_ci=True)
    fired_at = None
    recall_at_fire = None
    overlaps: list[float] = []

    for it in range(1, max_iter + 1):
        res = arms.expand(graph, seeds, gold, max_iter=it,
                          directions=("backward", "forward"))
        found = res.found if hasattr(res, "found") else res
        hit = gold & set(found)
        if not hit:
            continue
        n1, n2, m = arms.arm_counts(found, gold, "provs", "p0", "p1")
        total = m + (n1 - m) + (n2 - m)
        overlaps.append(m / total if total else 0.0)

        history.append(IterationStats(
            n=it, n_expanded=0, n_raw=0, n_after_dedup=0, n_eligible=len(found),
            n_screened=len(found), n_included=len(hit), n_excluded=len(found) - len(hit),
            cumulative_screened=len(found), cumulative_included=len(hit),
            frontier_size=len(hit),
        ))
        history.record_estimate(it, chapman(n1=n1, n2=n2, m=m))
        decision = rule.evaluate(history)
        if decision.triggered and fired_at is None:
            fired_at = it
            recall_at_fire = len(hit) / len(gold)

    final = arms.expand(graph, seeds, gold, max_iter=max_iter,
                        directions=("backward", "forward"))
    found_f = final.found if hasattr(final, "found") else final
    reachable = len(gold & set(found_f)) / len(gold)
    return dict(seed=seed, n_seeds=n_seeds,
                mean_overlap=St.mean(overlaps) if overlaps else 0.0,
                fired_at=fired_at, recall_at_fire=recall_at_fire,
                reachable_recall=reachable)


def run(quick: bool = False) -> dict:
    seeds = range(4 if quick else 12)
    sizes = (5,) if quick else (5, 10)
    out: dict[str, list[dict]] = {}
    for label, spec in (("optimistic", OPTIMISTIC), ("calibrated", CALIBRATED)):
        rows = []
        for s in seeds:
            for n in sizes:
                rows.append(_one_review(spec, s, n, max_iter=6))
        out[label] = rows
    return out


def summarise(out: dict) -> None:
    print(f"live arm overlap: {min(LIVE_OVERLAP.values()):.1%} to "
          f"{max(LIVE_OVERLAP.values()):.1%} across {len(LIVE_OVERLAP)} runs\n")
    print(f"{'setting':13s}{'reviews':>9s}{'arm overlap':>13s}{'rule fired':>12s}"
          f"{'recall at fire':>16s}{'reachable':>11s}")
    for label, rows in out.items():
        fired = [r for r in rows if r["fired_at"] is not None]
        print(f"{label:13s}{len(rows):>9d}{St.mean(r['mean_overlap'] for r in rows):>12.1%}"
              f"{len(fired):>7d}/{len(rows):<4d}"
              f"{(St.mean(r['recall_at_fire'] for r in fired) if fired else float('nan')):>16.3f}"
              f"{St.mean(r['reachable_recall'] for r in rows):>11.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = run(quick=args.quick)
    summarise(out)
    import json
    dest = HERE.parent / "cases" / "calibrated_stopping.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {dest.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
