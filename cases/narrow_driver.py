"""One iteration of the narrow-corpus run: expand, screen, label, evaluate.

Purpose: reach a RULE-DRIVEN stop on live data. The three earlier runs all stopped
on a screening budget, so the recall rule -- which needs the estimated population to
be stable across consecutive iterations -- never had a second estimate to compare
against and could not fire. This run is sized so three iterations fit inside a
budget comparable to one iteration of the earlier runs:

* six seeds with SHORT reference lists (15-33 refs), so iteration 1 is ~75 records
* backward direction only; forward expansion was 2-10x larger in every earlier run
  and is unbounded for a well-cited paper
* a generous screening budget that should NOT bind, so whichever rule fires is a
  saturation rule rather than a budget

Usage: python cases/narrow_driver.py init|step
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from snowballslr.config import Config  # noqa: E402
from snowballslr.core.run import Run  # noqa: E402

RUN = REPO / "run_narrow"
SEEDS = REPO / "cases" / "narrow_seeds.json"
STATE = Path("/tmp/narrow_cand.json")

CONFIG = {
    "run": {"name": "narrow-rule-driven", "max_iterations": 6, "directions": ["backward"]},
    "providers": {"order": ["openalex", "crossref"],
                  "openalex": {"enabled": True, "rps": 8},
                  "crossref": {"enabled": True, "rps": 8}},
    # OpenAlex changed its type vocabulary in July 2023; the library warns if the old
    # values are used, because they silently match nothing.
    "eligibility": {"types": ["article", "book-chapter"],
                    "require_title": True, "keep_unknown_metadata": True},
    "stopping": {"mode": "any_of",
                 "rules": [{"type": "marginal_yield", "eps": 0.02, "k": 2},
                           {"type": "estimated_recall", "tau": 0.95,
                            "min_stable_iterations": 2, "max_drift": 0.05},
                           # deliberately large: this must not be what fires
                           {"type": "budget", "max_screened": 20000},
                           {"type": "exhaustion"}]},
    "estimate": {"arms": [{"name": "openalex", "filter": {"provider": "openalex"}},
                          {"name": "crossref", "filter": {"provider": "crossref"}}],
                 "method": "chapman"},
}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step"
    if cmd == "init":
        seeds = [s["doi"] for s in json.loads(SEEDS.read_text())]
        cfg = Config.from_dict(CONFIG)
        for w in cfg.warnings():
            print(f"  config warning: {w}")
        run = Run.init(RUN, seeds=seeds, config=cfg)
        print(f"RESULT init: {len(run.state.works)} works, "
              f"{sum(1 for k in run.state.seeds if k in run.state.works)}/{len(seeds)} seeds resolved")
        cand = run.step()
        STATE.write_text(json.dumps([
            {"key": w.key, "title": w.title, "abstract": (w.abstract or "")[:1200],
             "year": w.year, "venue": w.venue,
             "cited_by_count": w.cited_by_count, "reference_count": w.reference_count}
            for w in cand], ensure_ascii=False))
        print(f"RESULT iter1: {len(cand)} candidates -> {STATE}")
        return 0

    run = Run.load(RUN)
    labels = REPO / f"labels_narrow_{run.state.iteration:03d}.csv"
    if not labels.exists():
        print(f"RESULT: no labels at {labels.name}; screen the candidates first")
        return 1
    run.label_from_file(labels)
    st = run.state
    fired = getattr(st, "stopped_by", None)
    print(f"RESULT labelled iter {st.iteration}: {len(st.decisions)} decisions, "
          f"phase={st.phase}, stopped_by={fired}")
    for d in (getattr(st, "last_decisions", None) or []):
        if isinstance(d, dict):
            dd = d
        elif hasattr(d, "to_dict"):
            dd = d.to_dict()
        else:
            dd = {"rule": str(d)}
        print(f"  rule {dd.get('rule')}: triggered={dd.get('triggered')} "
              f"value={dd.get('value')} - {(dd.get('rationale') or '')[:150]}")
    if st.phase == "stopped":
        print("RESULT: run stopped")
        return 0
    cand = run.step()
    STATE.write_text(json.dumps([
        {"key": w.key, "title": w.title, "abstract": (w.abstract or "")[:1200],
         "year": w.year, "venue": w.venue,
         "cited_by_count": w.cited_by_count, "reference_count": w.reference_count}
        for w in cand], ensure_ascii=False))
    print(f"RESULT iter{st.iteration + 1}: {len(cand)} candidates -> {STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
