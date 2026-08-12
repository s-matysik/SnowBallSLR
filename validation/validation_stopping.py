"""E3 -- stopping-rule benchmark on synthetic reviews with known ground truth.

For every (review, seed strategy, seed size) cell we run snowballing to
exhaustion, recording the full accumulation curve, then ask of each stopping
rule: at which iteration would it have fired, what true recall had been achieved
at that point, and what did stopping there cost or save relative to the
oracle-optimal stop (the earliest iteration reaching the target recall).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IterStat:
    n: int
    n_screened: int
    n_included: int
    cum_screened: int
    cum_included: int
    recall: float


def run_to_exhaustion(graph, seeds, gold, *, max_iter=12, providers=("p0","p1")) -> list[IterStat]:
    refs_p = graph["provider_edges"]
    fwd = {}
    for pj in providers:
        rev = {}
        for s, ts in refs_p[pj].items():
            for t in ts:
                rev.setdefault(t, []).append(s)
        fwd[pj] = rev

    decided = set(seeds)
    included = set(s for s in seeds if s in gold)
    frontier = list(seeds)
    cum_s = cum_i = 0
    out = []
    for i in range(1, max_iter + 1):
        if not frontier:
            break
        batch, frontier = frontier, []
        newly = set()
        for parent in batch:
            for pj in providers:
                for t in refs_p[pj].get(parent, []):
                    if t not in decided: newly.add(t)
                for s in fwd[pj].get(parent, []):
                    if s not in decided: newly.add(s)
        if not newly:
            break
        n_inc = 0
        for k in sorted(newly):
            decided.add(k)
            if k in gold:
                included.add(k); frontier.append(k); n_inc += 1
        cum_s += len(newly); cum_i += n_inc
        out.append(IterStat(i, len(newly), n_inc, cum_s, cum_i, len(included)/len(gold)))
    return out


# ---- stopping rules, reimplemented against the curve for scoring -------------

def rule_marginal_yield(hist, eps=0.01, k=2):
    """Fires when new includes per record screened stays below eps for k rounds."""
    streak = 0
    for s in hist:
        y = s.n_included / s.n_screened if s.n_screened else 0.0
        streak = streak + 1 if y < eps else 0
        if streak >= k:
            return s.n
    return None


def rule_exhaustion(hist):
    return hist[-1].n if hist else None


def rule_budget(hist, max_screened=2500):
    for s in hist:
        if s.cum_screened >= max_screened:
            return s.n
    return None


def rule_asymptote(hist, tau=0.95, min_points=4):
    """N(i) = N_inf (1 - exp(-lambda i)) fitted by grid+refine (no scipy dep)."""
    if len(hist) < min_points:
        return None
    for cut in range(min_points, len(hist) + 1):
        xs = [s.n for s in hist[:cut]]
        ys = [s.cum_included for s in hist[:cut]]
        n_obs = ys[-1]
        if n_obs <= 0:
            continue
        best, best_sse = None, float("inf")
        for n_inf in [n_obs * (1 + j / 40) for j in range(0, 81)]:
            for lam in [0.05 * j for j in range(1, 61)]:
                sse = sum((n_inf * (1 - math.exp(-lam * x)) - y) ** 2 for x, y in zip(xs, ys))
                if sse < best_sse:
                    best_sse, best = sse, (n_inf, lam)
        if best and best[0] > 0 and n_obs / best[0] >= tau:
            return hist[cut - 1].n
    return None


def rule_estimated_recall(hist, arm_hist_by_iter, tau=0.95, use_lower_ci=True):
    """arm_hist_by_iter: iteration -> (n1, n2, m, n_obs) over INCLUDED works."""
    from snowballslr.estimate.capture_recapture import chapman
    for s in hist:
        t = arm_hist_by_iter.get(s.n)
        if not t:
            continue
        n1, n2, m, n_obs = t
        est = chapman(n1, n2, m, n_observed=n_obs)
        if not est.estimable or est.recall is None:
            continue
        value = est.recall_ci[0] if (use_lower_ci and est.recall_ci) else est.recall
        if value >= tau:
            return s.n
    return None


def oracle_optimal(hist, target=0.95):
    """Earliest iteration reaching target recall; None if never reached."""
    for s in hist:
        if s.recall >= target:
            return s.n
    return None


def score(hist, fired, target=0.95):
    """Cost of stopping at `fired` relative to the oracle-optimal stop."""
    if not hist:
        return None
    opt = oracle_optimal(hist, target)
    final = hist[-1]
    if fired is None:
        return dict(fired_at=None, recall_at_stop=None, optimal=opt,
                    overshoot=None, excess_screened=None, missed=None, regret=None)
    st = next(s for s in hist if s.n == fired)
    ceiling = final.recall
    return dict(
        fired_at=fired,
        recall_at_stop=st.recall,
        optimal=opt,
        overshoot=(fired - opt) if opt else None,
        excess_screened=(st.cum_screened - next(s.cum_screened for s in hist if s.n == opt)) if opt else None,
        missed=round((ceiling - st.recall) * 0 + (final.cum_included - st.cum_included)),
        regret=ceiling - st.recall,
    )
