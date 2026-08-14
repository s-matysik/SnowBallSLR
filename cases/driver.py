"""One snowballing iteration in a single process: expand, then label.

Split across processes the pending state does not survive, so step() and label()
must run together. Invoked as:

    python cases/driver.py <case> init      # resolve seeds, first expansion
    python cases/driver.py <case> label     # ingest labels, evaluate rules
    python cases/driver.py <case> step      # expand the included frontier
    python cases/driver.py <case> report    # PRISMA, summary, verify
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from snowballslr.config import Config  # noqa: E402
from snowballslr.core.run import Run  # noqa: E402

CASES = json.loads((REPO / "cases" / "seed_selection.json").read_text())


def config_for(case: str) -> Config:
    info = CASES[case]
    return Config.from_dict({
        "run": {"directions": ["backward", "forward"], "max_iterations": 4},
        "providers": {"order": ["openalex", "crossref"],
                      "openalex": {"enabled": True}, "crossref": {"enabled": True},
                      "semanticscholar": {"enabled": False}, "grobid": {"enabled": False}},
        # OpenAlex reports journal, conference and preprint items under "article"
        # since July 2023; the pre-2023 vocabulary matches nothing.
        "eligibility": {"types": ["article", "book-chapter"]},
        "stopping": {
            "mode": "any_of",
            "rules": [
                {"type": "estimated_recall", "tau": 0.95, "use_lower_ci": True},
                {"type": "marginal_yield", "eps": 0.01, "k": 2},
                {"type": "budget", "max_screened": info["budget"]},
                {"type": "exhaustion"},
            ],
        },
        "estimate": {
            "method": "loglinear",
            "arms": [
                {"name": "openalex", "filter": {"provider": "openalex"}},
                {"name": "crossref", "filter": {"provider": "crossref"}},
                {"name": "scopus", "filter": {"member_of": "scopus"}},
            ],
            "membership_sets": {"scopus": info["export"]},
        },
        "determinism": {"cache_dir": "cache"},
    })


def emit(run: Run, extra: dict | None = None) -> None:
    out = {
        "iteration": run.state.iteration,
        "phase": str(run.state.phase),
        "works": len(run.state.works),
        "pending": len(run.state.pending),
        "decisions": len(run.state.decisions),
        "included": len(run.state.included),
        "stopped_by": run.state.stopped_by,
    }
    out.update(extra or {})
    print("RESULT " + json.dumps(out))


def main() -> None:
    case, action = sys.argv[1], sys.argv[2]
    root = REPO / f"run_{case}"
    cfg = config_for(case)
    for w in cfg.warnings():
        print("CONFIG WARNING:", w)

    if action == "init":
        dois = [s["doi"] for s in CASES[case]["seeds"]]
        run = Run.init(root, seeds=dois, config=cfg)
        cand = run.step()
        (root / "candidates.json").write_text(json.dumps(
            [{"key": w.key, "title": w.title, "year": w.year,
              "abstract": (w.abstract or "")[:1500], "unresolved": w.unresolved}
             for w in cand], ensure_ascii=False))
        emit(run, {"candidates": len(cand),
                   "seeds_resolved": sum(1 for k in run.state.seeds if k in run.state.works)})
        return

    run = Run.load(root)
    if action == "label":
        run.label_from_file(REPO / f"labels_{case}_{run.state.iteration:03d}.csv")
        # state.last_decisions holds dicts (StopDecision.to_dict()), not objects.
        fired = [{"rule": d.get("rule"), "triggered": d.get("triggered"),
                  "value": d.get("value"), "rationale": (d.get("rationale") or "")[:200]}
                 for d in (run.state.last_decisions or [])]
        emit(run, {"rules": fired})
        return

    if action == "step":
        cand = run.step()
        (root / "candidates.json").write_text(json.dumps(
            [{"key": w.key, "title": w.title, "year": w.year,
              "abstract": (w.abstract or "")[:1500], "unresolved": w.unresolved}
             for w in cand], ensure_ascii=False))
        emit(run, {"candidates": len(cand)})
        return

    if action == "report":
        paths = run.report(prisma=True, graph=True)
        est = run.estimate_recall()
        # RecallEstimateRecord is the flattened per-iteration record: it carries
        # n/n_hat/recall directly, not a nested `estimate` and not `iteration`.
        trail = [{"iteration": i + 1,
                  "n": getattr(r, "n", None),
                  "estimable": getattr(r, "estimable", None),
                  "n_hat": getattr(r, "n_hat", None),
                  "recall": getattr(r, "recall", None),
                  "recall_lower": getattr(r, "recall_lower", None)}
                 for i, r in enumerate(run.state.history.recall_estimate_trail or [])]
        emit(run, {
            "artifacts": {k: str(v) for k, v in paths.items()},
            "estimate": {"method": est.method, "estimable": est.estimable,
                         "n_hat": est.n_hat, "n_hat_ci": est.n_hat_ci,
                         "recall": est.recall, "recall_ci": est.recall_ci,
                         "reason": est.reason,
                         "warnings": list(est.warnings)[:6],
                         "detail": {k: v for k, v in est.detail.items() if k != "assumptions"}},
            "trail": trail,
        })
        return

    raise SystemExit(f"unknown action {action!r}")


if __name__ == "__main__":
    main()
