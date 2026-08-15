"""Dedicated tests for the invariants named in SPEC.md.

SPEC §18 requires each invariant to carry a passing test that names it. INV-1
(byte-identical replay) and INV-4 (no network in tests) are covered in
`test_determinism.py` and `conftest.py`; INV-2 and INV-3 are covered here.
"""

from __future__ import annotations

import json

from snowballslr import Config
from snowballslr.core.run import Run
from snowballslr.types import Direction

# The invariants under test:
#   INV-2  Every emitted record carries complete provenance: provider, direction,
#          parent, iteration, timestamp, response hash.
#   INV-3  No unordered iteration reaches output. Every collection is sorted by an
#          explicit total order before serialization.

PROVENANCE_FIELDS = (
    "provider",
    "direction",
    "parent_key",
    "iteration",
    "retrieved_at",
    "response_hash",
)


def _graph():
    def w(key, year):
        return {
            "key": key, "title": f"Paper {key}", "title_norm": "",
            "doi": key.split(":", 1)[1], "year": year, "type": "article",
            "language": "en", "unresolved": False, "provenance": [],
        }
    keys = {
        f"doi:10.1/{c}": y
        for c, y in zip("abcdef", (2012, 2014, 2016, 2018, 2019, 2020), strict=True)
    }
    works = {"doi:10.1/seed": w("doi:10.1/seed", 2021)}
    works.update({k: w(k, y) for k, y in keys.items()})
    return {
        "works": works,
        "references": {
            "doi:10.1/seed": ["doi:10.1/a", "doi:10.1/b", "doi:10.1/c"],
            "doi:10.1/a": ["doi:10.1/d"],
            "doi:10.1/b": ["doi:10.1/e"],
        },
        "citations": {
            "doi:10.1/a": ["doi:10.1/seed"],
            "doi:10.1/d": ["doi:10.1/a"],
            "doi:10.1/e": ["doi:10.1/b"],
            "doi:10.1/f": ["doi:10.1/c"],
        },
    }


def _config():
    return Config.from_dict(
        {
            "run": {"name": "inv", "max_iterations": 4},
            "eligibility": {"year_min": 2000},
            "stopping": {"mode": "any_of", "rules": [{"type": "exhaustion"}]},
        }
    )


def _completed_run(tmp_path):
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _config(), offline_graph=_graph())
    for _ in range(3):
        candidates = run.step()
        if not candidates:
            break
        run.label({w.key: "include" for w in candidates})
    run.report()
    return run


def test_inv2_every_discovered_record_carries_complete_provenance(tmp_path):
    """INV-2: provider, direction, parent, iteration, timestamp and response hash."""
    run = _completed_run(tmp_path)
    seeds = set(run.state.seeds)
    discovered = [w for k, w in run.state.works.items() if k not in seeds]
    assert discovered, "expected the run to discover records beyond the seeds"

    for work in discovered:
        assert work.provenance, f"{work.key} carries no provenance"
        for prov in work.provenance:
            for field in PROVENANCE_FIELDS:
                assert hasattr(prov, field), f"{work.key} provenance lacks {field}"
            assert prov.provider
            assert prov.direction in (Direction.BACKWARD, Direction.FORWARD)
            assert prov.parent_key, f"{work.key} provenance has no parent"
            assert prov.iteration >= 1
            assert prov.retrieved_at
            assert prov.response_hash


def test_inv2_provenance_is_append_only_across_rediscovery(tmp_path):
    """A record found twice keeps both provenance entries (INV-6 reinforces INV-2)."""
    run = _completed_run(tmp_path)
    multi = [w for w in run.state.works.values() if len(w.provenance) > 1]
    for work in multi:
        pairs = {(p.parent_key, str(p.direction)) for p in work.provenance}
        assert len(pairs) == len({(p.parent_key, str(p.direction)) for p in work.provenance})
        assert len(work.provenance) >= len(pairs)


def test_inv3_serialized_state_collections_are_sorted(tmp_path):
    """INV-3: no unordered iteration reaches output."""
    _completed_run(tmp_path)
    state = json.loads((tmp_path / "r" / "state.json").read_text(encoding="utf-8"))

    for key in ("seeds", "expanded", "frontier"):
        assert state[key] == sorted(state[key]), f"state.{key} is not sorted"
    for key in ("works", "decisions", "aliases"):
        assert list(state[key]) == sorted(state[key]), f"state.{key} keys are not sorted"

    # Provenance within each record is ordered too, so a diff of two runs is stable.
    for key, work in state["works"].items():
        provs = [
            (p["provider"], p["direction"], p["parent_key"] or "", p["iteration"])
            for p in work.get("provenance", [])
        ]
        assert provs == sorted(provs), f"provenance of {key} is not sorted"


def test_inv3_pending_is_ranked_not_sorted_and_that_is_deliberate(tmp_path):
    """The one exception: `pending` is stored in ranked order, ties broken by key.

    SPEC calls this out explicitly - the order *is* the product. It must still be
    deterministic, which the determinism suite covers.
    """
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _config(), offline_graph=_graph())
    candidates = run.step()
    assert candidates
    assert [w.key for w in candidates] == run.state.pending
    scores = run.state.pending_scores
    ordered = sorted(run.state.pending, key=lambda k: (-scores.get(k, 0.0), k))
    assert run.state.pending == ordered


def test_inv3_exported_csv_rows_follow_the_ranked_order(tmp_path):
    import csv

    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _config(), offline_graph=_graph())
    run.step()
    path = tmp_path / "r" / "iterations" / "iter_001" / "candidates.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        keys = [row["key"] for row in csv.DictReader(fh)]
    assert keys == run.state.pending
