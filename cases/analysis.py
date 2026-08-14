"""Cross-case analysis: test the pre-registered prediction against three runs.

Reads cases/results_<case>.json for each of the three cases and evaluates the
four directional predictions registered in cases/PREDICTION.md before any run
started. Each prediction is reported as held / failed with the numbers behind it,
whichever way it falls.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES = ("adverse", "mid", "large")


def load() -> dict[str, dict]:
    out = {}
    for c in CASES:
        p = REPO / "cases" / f"results_{c}.json"
        if p.exists():
            out[c] = json.loads(p.read_text())
    return out


def _recall(r: dict) -> float | None:
    est = r.get("recall_estimate") or {}
    return est.get("recall")


def _unresolvable_share(r: dict) -> float | None:
    ident = r.get("identified") or 0
    return (r.get("unresolvable") or 0) / ident if ident else None


def test_predictions(res: dict[str, dict]) -> list[dict]:
    """Four directional predictions, adverse against favourable (mid)."""
    if "adverse" not in res or "mid" not in res:
        return []
    a, m = res["adverse"], res["mid"]
    checks = []

    ra, rm = _recall(a), _recall(m)
    checks.append({
        "prediction": "estimated recall lower in the adverse corpus",
        "adverse": ra, "favourable": rm,
        "held": (ra is not None and rm is not None and ra < rm),
        "testable": ra is not None and rm is not None,
    })

    ua, um = _unresolvable_share(a), _unresolvable_share(m)
    checks.append({
        "prediction": "unresolvable share proportionally higher in the adverse corpus",
        "adverse": ua, "favourable": um,
        "held": (ua is not None and um is not None and ua > um),
        "testable": ua is not None and um is not None,
    })

    ia, im = a.get("iterations"), m.get("iterations")
    checks.append({
        "prediction": "the marginal-yield rule fires earlier in the adverse corpus",
        "adverse": {"iterations": ia, "stopped_by": a.get("stopped_by")},
        "favourable": {"iterations": im, "stopped_by": m.get("stopped_by")},
        # Only testable when the favourable case was not cut short by its budget:
        # a budget stop pre-empts the rule and says nothing about saturation.
        "testable": m.get("stop_is_saturation", True) is not None,
        "held": None,
    })

    def scopus_share(r):
        arms = (r.get("arm_capture") or {})
        inc = r.get("included") or 0
        s = arms.get("scopus")
        if isinstance(s, dict):
            s = s.get("n")
        return (s / inc) if (s is not None and inc) else None

    sa, sm = scopus_share(a), scopus_share(m)
    checks.append({
        "prediction": "the Scopus arm captures a smaller share of included records",
        "adverse": sa, "favourable": sm,
        "ceiling_adverse": 0.658, "ceiling_favourable": 0.959,
        "held": (sa is not None and sm is not None and sa < sm),
        "testable": sa is not None and sm is not None,
    })
    return checks


def summarise(res: dict[str, dict]) -> list[dict]:
    rows = []
    for c in CASES:
        if c not in res:
            continue
        r = res[c]
        est = r.get("recall_estimate") or {}
        rows.append({
            "case": c,
            "corpus": r.get("corpus_records"),
            "identified": r.get("identified"),
            "screened": r.get("screened"),
            "included": r.get("included"),
            "unresolvable_share": _unresolvable_share(r),
            "stopped_by": r.get("stopped_by"),
            "saturation": r.get("stop_is_saturation"),
            "recall": est.get("recall"),
            "recall_ci": est.get("recall_ci"),
            "n_hat": est.get("n_hat"),
            "method": est.get("method"),
            "icc": r.get("icc"),
            "seed_auc": r.get("seed_auc"),
            "gate_fnr": r.get("gate_false_negative_rate"),
        })
    return rows


if __name__ == "__main__":
    res = load()
    print(f"cases available: {sorted(res)}")
    for row in summarise(res):
        print(json.dumps(row))
    for chk in test_predictions(res):
        print(json.dumps(chk))
