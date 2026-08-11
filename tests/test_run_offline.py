"""End-to-end runs against the frozen fixture graph."""

from __future__ import annotations

import pytest

from snowballslr import Config, Run
from snowballslr.core.state import Phase
from snowballslr.errors import LabelError, StateError


def _drive(run, oracle, max_iter=8):
    while not run.stopped and run.state.iteration < max_iter:
        candidates = run.step()
        if not candidates:
            break
        run.label({w.key: oracle.decide(w.key) for w in candidates})
    return run


def test_seeds_resolve_and_enter_the_frontier(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    assert len(run.state.seeds) == len(seeds)
    assert set(run.state.frontier) == set(run.state.seeds)
    assert all(run.state.decision_for(k) == "include" for k in run.state.seeds)


def test_run_reaches_high_recall_and_stops(tmp_path, graph, seeds, config, oracle):
    run = _drive(Run.init(tmp_path / "r", seeds, config, offline_graph=graph), oracle)
    found = set(run.state.included_keys) & oracle.included
    assert len(found) / len(oracle) >= 0.90
    assert run.stopped and run.state.stopped_by


def test_step_is_idempotent_without_labels(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    first = [w.key for w in run.step()]
    second = [w.key for w in run.step()]
    assert first == second
    assert run.state.phase == Phase.AWAITING_LABELS


def test_label_rejects_unknown_keys(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    run.step()
    with pytest.raises(LabelError):
        run.label({"doi:10.9999/nonexistent": "include"})


def test_label_rejects_invalid_decisions(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    candidates = run.step()
    with pytest.raises(LabelError):
        run.label({candidates[0].key: "maybe"})


def test_label_before_step_is_a_state_error(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    with pytest.raises(StateError):
        run.label({})


def test_run_resumes_from_disk(tmp_path, graph, seeds, config, oracle):
    root = tmp_path / "r"
    run = Run.init(root, seeds, config, offline_graph=graph)
    candidates = run.step()
    run.label({w.key: oracle.decide(w.key) for w in candidates})
    included_before = list(run.state.included_keys)

    resumed = Run.load(root, offline_graph=graph, config=config)
    assert resumed.state.included_keys == included_before
    assert resumed.state.iteration == 1
    _drive(resumed, oracle)
    assert resumed.stopped


def test_only_included_studies_extend_the_frontier(tmp_path, graph, seeds, config, oracle):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    candidates = run.step()
    run.label({w.key: oracle.decide(w.key) for w in candidates})
    assert set(run.state.frontier) <= set(run.state.included_keys)


def test_eligibility_filter_excludes_out_of_range_years(tmp_path, graph, seeds):
    cfg = Config.from_dict(
        {"run": {"max_iterations": 2}, "eligibility": {"year_min": 2050}}
    )
    run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    assert run.step() == []


def test_max_iterations_is_respected(tmp_path, graph, seeds, oracle):
    cfg = Config.from_dict(
        {
            "run": {"max_iterations": 1},
            "stopping": {"mode": "any_of", "rules": [{"type": "exhaustion"}]},
        }
    )
    run = _drive(Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph), oracle, max_iter=5)
    assert len(run.state.history) <= 1


def test_simulation_loop_matches_manual_driving(tmp_path, graph, seeds, config, oracle):
    manual = _drive(Run.init(tmp_path / "a", seeds, config, offline_graph=graph), oracle)
    auto = Run.init(tmp_path / "b", seeds, config, offline_graph=graph)
    auto.run_to_saturation(oracle.as_mapping())
    assert auto.state.included_keys == manual.state.included_keys


def test_reports_are_written(tmp_path, graph, seeds, config, oracle):
    run = _drive(Run.init(tmp_path / "r", seeds, config, offline_graph=graph), oracle)
    produced = run.report(prisma=True, graph=True)
    for name in ("prisma.json", "prisma.svg", "network.graphml", "report.md", "included.csv"):
        assert produced[name].exists() and produced[name].stat().st_size > 0
    assert "<svg" in produced["prisma.svg"].read_text(encoding="utf-8")


def test_audit_trail_records_expansion_and_stop_evaluation(tmp_path, graph, seeds, config, oracle):
    run = _drive(Run.init(tmp_path / "r", seeds, config, offline_graph=graph), oracle)
    events = {e["event"] for e in run.audit.read()}
    assert {"seed_resolved", "expand", "stop_eval", "stopped"} <= events


def test_every_rule_is_evaluated_every_iteration(tmp_path, graph, seeds, config, oracle):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    candidates = run.step()
    run.label({w.key: oracle.decide(w.key) for w in candidates})
    evaluated = {d["rule"] for d in run.state.last_decisions}
    assert evaluated == {"marginal_yield", "budget", "exhaustion"}


# -- regression: silent over-filtering must be visible ---------------------


def test_filter_report_names_the_reason_and_the_values_seen(tmp_path, graph, seeds):
    """A type vocabulary mismatch removed an entire real corpus once. Never silently again."""
    cfg = Config.from_dict(
        {
            "run": {"max_iterations": 2},
            "eligibility": {"types": ["journal-article"]},  # legacy OpenAlex value
        }
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    assert run.step() == []

    report = run.last_rejection_report
    assert report["n_rejected"] == report["n_fresh"] > 0
    assert report["share_rejected"] == 1.0
    assert report["by_reason"] == {"type_not_allowed": report["n_rejected"]}
    # The values actually observed are what make the mismatch diagnosable.
    assert "journal-article" not in report["observed_values"]["type"]
    assert sum(report["observed_values"]["type"].values()) == report["n_rejected"]


def test_filter_report_is_empty_when_nothing_is_filtered(tmp_path, graph, seeds, config):
    run = Run.init(tmp_path / "r", seeds, config, offline_graph=graph)
    run.step()
    assert not run.last_rejection_report.get("n_rejected")


def test_ineligibility_reasons_are_specific(tmp_path, graph, seeds):
    from snowballslr.types import Work

    cfg = Config.from_dict(
        {
            "eligibility": {
                "year_min": 2015,
                "year_max": 2020,
                "require_abstract": True,
                "keep_unknown_metadata": False,
            }
        }
    )
    run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    def mk(**kw):
        base = {
            "key": "k", "title": "T", "title_norm": "t",
            "abstract": "a", "year": 2018,
        }
        return Work(**{**base, **kw})

    assert run._ineligibility_reason(mk()) is None
    assert run._ineligibility_reason(mk(year=2010)) == "year_below_min"
    assert run._ineligibility_reason(mk(year=2030)) == "year_above_max"
    assert run._ineligibility_reason(mk(year=None)) == "no_year"
    assert run._ineligibility_reason(mk(is_retracted=True)) == "retracted"
    assert run._ineligibility_reason(Work(key="k", title="", title_norm="")) == "no_title"
    assert (
        run._ineligibility_reason(Work(key="k", title="T", title_norm="t", year=2018))
        == "no_abstract"
    )


def test_missing_metadata_is_kept_by_default(tmp_path, graph, seeds):
    """Absent metadata is not evidence of ineligibility -- and the worst-indexed
    records are precisely the ones citation searching exists to recover."""
    from snowballslr.types import Work

    cfg = Config.from_dict(
        {"eligibility": {"year_min": 2015, "types": ["article"], "languages": ["en"]}}
    )
    run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    unknown = Work(key="k", title="T", title_norm="t")  # no year, type or language
    assert run._ineligibility_reason(unknown) is None


def test_strict_mode_rejects_missing_metadata_with_named_reasons(tmp_path, graph, seeds):
    from snowballslr.types import Work

    cfg = Config.from_dict(
        {
            "eligibility": {
                "year_min": 2015,
                "types": ["article"],
                "languages": ["en"],
                "keep_unknown_metadata": False,
            }
        }
    )
    run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    assert run._ineligibility_reason(Work(key="k", title="T", title_norm="t")) == "no_year"
    assert (
        run._ineligibility_reason(Work(key="k", title="T", title_norm="t", year=2020))
        == "no_type"
    )
    assert (
        run._ineligibility_reason(
            Work(key="k", title="T", title_norm="t", year=2020, type="article")
        )
        == "no_language"
    )


def test_out_of_range_year_is_still_rejected_when_known(tmp_path, graph, seeds):
    from snowballslr.types import Work

    cfg = Config.from_dict({"eligibility": {"year_min": 2015, "year_max": 2020}})
    run = Run.init(tmp_path / "r", seeds, cfg, offline_graph=graph)
    def mk(y):
        return Work(key="k", title="T", title_norm="t", year=y)

    assert run._ineligibility_reason(mk(2010)) == "year_below_min"
    assert run._ineligibility_reason(mk(2030)) == "year_above_max"
    assert run._ineligibility_reason(mk(2018)) is None
