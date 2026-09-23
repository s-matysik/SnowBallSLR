"""Regression tests for defects found during the SoftwareX validation study.

Each test names the failure it locks down. All are offline and deterministic.
"""

from __future__ import annotations

import re

import pytest

from snowballslr import Config
from snowballslr.core.run import Run
from snowballslr.errors import LabelError
from snowballslr.identity.dedup import blocking_keys, deduplicate
from snowballslr.identity.keys import canonical_key
from snowballslr.identity.normalize import normalize_title
from snowballslr.types import Author, Work

# -- identity layer ---------------------------------------------------------


def _work(title, year, surname, doi=None, key=None, unresolved=False):
    k = key or canonical_key(doi=doi, title=title, year=year, first_author_surname=surname)
    return Work(
        key=k,
        title=title,
        title_norm=normalize_title(title),
        doi=doi,
        year=year,
        authors=(Author(surname=surname),) if surname else (),
        unresolved=unresolved,
    )


def test_year_bucket_boundary_no_longer_hides_duplicates():
    """1995/1996 fell in different `year // 2` buckets and were never compared,
    despite tiers T2/T3 allowing a +/-1 year tolerance."""
    a = _work("Assistive Technologies Principles and Practice", 1995, "Cook", doi="10.1/a")
    b = _work("Assistive Technologies Principles and Practice", 1996, "Cook")
    assert set(blocking_keys(a)) & set(blocking_keys(b))
    assert len(deduplicate([a, b]).works) == 1


def test_initials_prefix_surname_matches_bare_surname():
    """Crossref unstructured references yield "AM Cook" where OpenAlex yields "Cook"."""
    a = _work("Assistive Technologies Principles and Practice", 1995, "Cook", doi="10.1/a")
    b = _work("Assistive Technologies Principles and Practice", 1995, "AM Cook")
    assert len(deduplicate([a, b]).works) == 1


def test_surname_suffix_match_does_not_over_merge():
    """A shared suffix that is not an initials prefix must NOT merge."""
    a = _work("A Study of Something Particular", 2010, "Berg", doi="10.1/a")
    b = _work("A Study of Something Particular", 2010, "Vandenberg")
    assert len(deduplicate([a, b]).works) == 2


def test_works_without_a_year_share_a_block():
    a = _work("Some Untitled Reference String", None, "Smith")
    b = _work("Some Untitled Reference String", None, "Smith")
    assert set(blocking_keys(a)) & set(blocking_keys(b))


def test_dedup_remains_order_independent_with_multi_key_blocking():
    works = [
        _work("Assistive Technologies Principles and Practice", 1995, "Cook", doi="10.1/a"),
        _work("Assistive Technologies Principles and Practice", 1996, "AM Cook"),
        _work("An Entirely Different Paper About Other Things", 2001, "Jones", doi="10.1/b"),
    ]
    base = [w.key for w in deduplicate(works).works]
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        assert [w.key for w in deduplicate([works[i] for i in perm]).works] == base


# -- run loop ---------------------------------------------------------------


def _tiny_graph():
    """Seed cites A and B; A cites C. All resolvable."""
    def w(key, year):
        return {
            "key": key, "title": f"Paper {key}", "title_norm": "",
            "doi": key.split(":", 1)[1], "year": year, "type": "article",
            "language": "en", "unresolved": False, "provenance": [],
        }
    return {
        "works": {
            "doi:10.1/seed": w("doi:10.1/seed", 2020),
            "doi:10.1/a": w("doi:10.1/a", 2015),
            "doi:10.1/b": w("doi:10.1/b", 2016),
            "doi:10.1/c": w("doi:10.1/c", 2012),
        },
        "references": {"doi:10.1/seed": ["doi:10.1/a", "doi:10.1/b"], "doi:10.1/a": ["doi:10.1/c"]},
        "citations": {"doi:10.1/a": ["doi:10.1/seed"], "doi:10.1/c": ["doi:10.1/a"]},
    }


def _cfg():
    return Config.from_dict(
        {
            "run": {"name": "reg", "max_iterations": 4},
            "eligibility": {"year_min": 2000},
            "stopping": {"mode": "any_of", "rules": [{"type": "exhaustion"}]},
            "estimate": {
                "arms": [
                    {"name": "openalex", "filter": {"provider": "openalex"}},
                    {"name": "crossref", "filter": {"provider": "crossref"}},
                ]
            },
        }
    )


def test_labelling_nothing_is_refused_instead_of_stopping_by_exhaustion(tmp_path):
    """A labels file with every decision cell blank previously terminated the run
    via `exhaustion`, reporting a completed run that screened zero records."""
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _cfg(), offline_graph=_tiny_graph())
    candidates = run.step()
    assert candidates
    with pytest.raises(LabelError, match="none of the"):
        run.label({w.key: "unscreened" for w in candidates})
    assert run.state.phase != "stopped"


def test_empty_labelling_is_allowed_when_explicitly_requested(tmp_path):
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _cfg(), offline_graph=_tiny_graph())
    candidates = run.step()
    run.label({w.key: "unscreened" for w in candidates}, allow_empty=True)
    assert run.state.stopped_by == "exhaustion"


def test_unresolved_records_are_never_expanded(tmp_path):
    """README promises unresolved records are "never expanded forward"; the core
    loop did not enforce it, so an included unresolved record was expanded."""
    graph = _tiny_graph()
    graph["works"]["doi:10.1/a"]["unresolved"] = True
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _cfg(), offline_graph=graph)
    candidates = run.step()
    run.label({w.key: "include" for w in candidates})
    before = set(run.state.works)
    run.step()
    # doi:10.1/c is reachable only by expanding the unresolved doi:10.1/a.
    assert "doi:10.1/c" not in set(run.state.works) - before


def test_recall_estimate_trail_survives_save_and_load(tmp_path):
    """The closure guard reads the trail, so it must persist across resume."""
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _cfg(), offline_graph=_tiny_graph())
    candidates = run.step()
    run.label({w.key: "include" for w in candidates})
    trail = run.state.history.recall_estimate_trail
    assert trail
    reloaded = Run.load(tmp_path / "r", offline_graph=_tiny_graph())
    assert [r.to_dict() for r in reloaded.state.history.recall_estimate_trail] == [
        r.to_dict() for r in trail
    ]


# -- configuration ----------------------------------------------------------


def test_direction_arms_warning_names_the_remedy():
    """Direction arms were the shipped default and failed silently: the estimate
    was simply refused for insufficient overlap, with nothing pointing at why."""
    cfg = Config.from_dict(
        {
            "estimate": {
                "arms": [
                    {"name": "backward", "filter": {"direction": "backward"}},
                    {"name": "forward", "filter": {"direction": "forward"}},
                ]
            }
        }
    )
    (warning,) = [w for w in cfg.warnings() if "direction" in w]
    assert "provider arms" in warning


# -- estimators -------------------------------------------------------------


def test_chao1_refuses_when_every_record_was_seen_once():
    """Chao1 previously returned estimable=True with N-hat = 114,481 (recall
    0.4%) on a real 478-record run whose arms were disjoint."""
    from snowballslr.estimate.chao import chao1

    est = chao1({f"k{i}": ["backward"] for i in range(478)})
    assert not est.estimable
    assert "doubletons" in est.reason and est.n_hat is None


def test_chao1_still_estimates_when_doubletons_exist():
    from snowballslr.estimate.chao import chao1

    est = chao1({f"k{i}": (["a", "b"] if i < 30 else ["a"]) for i in range(100)})
    assert est.estimable and est.n_hat > est.n_observed


def test_chao1_still_estimates_when_records_were_seen_three_times():
    """f2 == 0 is only degenerate when it coincides with f1 == S_obs."""
    from snowballslr.estimate.chao import chao1

    est = chao1({f"k{i}": (["a", "b", "c"] if i < 5 else ["a"]) for i in range(50)})
    assert est.estimable


# -- verification -----------------------------------------------------------


def _run_with_cache(tmp_path):
    """A completed run whose cache holds one real entry.

    The offline provider replays a frozen graph and never writes to the cache,
    so the entry is written through the Cache API itself -- which is the surface
    `verify` inspects.
    """
    run = Run.init(tmp_path / "r", ["doi:10.1/seed"], _cfg(), offline_graph=_tiny_graph())
    candidates = run.step()
    run.label({w.key: "include" for w in candidates})
    run.report()

    from snowballslr.determinism.cache import Cache

    cache = Cache(tmp_path / "r" / "cache")
    cache.get_or_fetch(
        "openalex", "references", {"id": "W1"},
        lambda: ({"results": [{"id": "W2"}]}, 200, {"api": "v1"}),
    )
    return run


def _sole_cache_entry(tmp_path):
    entries = sorted((tmp_path / "r" / "cache").rglob("*.json"))
    assert entries, "expected a cache entry to verify against"
    return entries[0]


def test_verify_detects_a_tampered_cache_body(tmp_path):
    """`verify` counted cache entries but never checked them, so an edited cache
    body passed with ok=True."""
    import json as _json

    from snowballslr.determinism.verify import verify_replay

    _run_with_cache(tmp_path)
    assert verify_replay(tmp_path / "r", strict=False).ok

    entry = _sole_cache_entry(tmp_path)
    payload = _json.loads(entry.read_text())
    payload["body"] = {"tampered": True}
    entry.write_text(_json.dumps(payload))

    report = verify_replay(tmp_path / "r", strict=False)
    assert not report.ok
    assert any("recorded hash" in m for m in report.cache_entries_mismatched)


def test_verify_detects_a_tampered_request_key(tmp_path):
    import json as _json

    from snowballslr.determinism.verify import verify_replay

    _run_with_cache(tmp_path)
    entry = _sole_cache_entry(tmp_path)
    payload = _json.loads(entry.read_text())
    payload["params"] = {**payload.get("params", {}), "sneaky": 1}
    entry.write_text(_json.dumps(payload))

    report = verify_replay(tmp_path / "r", strict=False)
    assert not report.ok
    assert any("request key" in m for m in report.cache_entries_mismatched)

def test_report_quotes_the_manifest_config_hash_not_the_live_one(tmp_path):
    """The report printed `run.config.config_hash`, the LIVE value. Editing the config
    after a run -- raising an iteration cap to continue past it, say -- made the report
    quote a hash the manifest could not corroborate, while claiming the reader could
    check it against that manifest."""
    run = _run_with_cache(tmp_path)
    recorded = run.manifest.config_hash

    # change the config the way continuing a stopped run does, then regenerate
    run.config.run.max_iterations += 6
    assert run.config.config_hash != recorded, "expected the edit to move the hash"
    run.report()

    report = (tmp_path / "r" / "outputs" / "report.md").read_text()
    hashes = re.findall(r"sha256:[0-9a-f]{64}", report)
    assert hashes[0] == recorded, "the first hash printed must be the manifest's own"
    assert "Configuration changed after this run was written" in report
    assert run.config.config_hash in report, "the divergent live hash must be named too"


def test_report_omits_the_drift_note_when_config_is_unchanged(tmp_path):
    """The disclosure must not fire on a clean run, or every report carries a warning."""
    run = _run_with_cache(tmp_path)
    run.report()
    report = (tmp_path / "r" / "outputs" / "report.md").read_text()
    assert run.manifest.config_hash in report
    assert "Configuration changed" not in report


def test_verify_detects_a_config_edited_after_the_run(tmp_path):
    """The drift check must read the LIVE config file, not the manifest's own snapshot.

    A first version compared the snapshot against itself, which passes by construction
    and therefore missed the case it was written for: an iteration cap raised in
    `<run_dir>/config.yaml` to continue a stopped run, after which any regenerated
    report quotes a hash the manifest cannot corroborate.
    """
    from snowballslr.determinism.verify import verify_replay

    run = _run_with_cache(tmp_path)
    assert verify_replay(tmp_path / "r", strict=False).ok

    run.config.run.max_iterations += 6
    run.config.to_yaml(tmp_path / "r" / "config.yaml")

    report = verify_replay(tmp_path / "r", strict=False)
    assert not report.ok
    assert report.config_drift is not None
    assert "run.max_iterations" in report.config_drift
    assert "config.yaml" in report.config_drift


def test_verify_rejects_a_manifest_whose_snapshot_contradicts_its_hash(tmp_path):
    """`verify` never rehashed the manifest's own config snapshot, so a snapshot edited
    apart from its recorded hash passed unnoticed."""
    import json as _json

    from snowballslr.determinism.verify import verify_replay

    _run_with_cache(tmp_path)
    assert verify_replay(tmp_path / "r", strict=False).ok

    manifest_path = tmp_path / "r" / "run.json"
    payload = _json.loads(manifest_path.read_text())
    payload["config"]["stopping"]["mode"] = "all_of"
    manifest_path.write_text(_json.dumps(payload))

    report = verify_replay(tmp_path / "r", strict=False)
    assert not report.ok
    assert report.config_drift is not None
    assert "altered apart from its hash" in report.config_drift


def test_verify_tolerates_a_setting_added_after_the_run(tmp_path):
    """A run predating a later-added setting stores no value for it, so reloading fills
    in that setting's default and moves the hash. Failing on that would make `verify`
    useless against any run older than the current release."""
    import json as _json

    from snowballslr.determinism.verify import verify_replay

    _run_with_cache(tmp_path)
    manifest_path = tmp_path / "r" / "run.json"
    payload = _json.loads(manifest_path.read_text())
    # emulate an older manifest: the settings simply are not recorded at all
    payload["config"]["estimate"].pop("membership_doi_column", None)
    payload["config"]["estimate"].pop("membership_sets", None)
    manifest_path.write_text(_json.dumps(payload))

    report = verify_replay(tmp_path / "r", strict=False)
    assert report.config_drift is None, report.config_drift
    assert report.ok


def test_chapman_refuses_nested_arms_rather_than_reporting_recall_one():
    """R2 revision: nested arms must be refused, not returned as recall 1.0.

    With m == min(n1, n2) one arm is contained in the other, Chapman's point
    estimate cannot exceed the observed count, and the previous implementation
    returned it as estimable with recall exactly 1.0 and an interval whose lower
    bound (clamped to the observed count) could exceed its upper bound.
    """
    from snowballslr.estimate.capture_recapture import chapman

    est = chapman(337, 83, 83, n_observed=337)
    assert est.estimable is False
    assert est.n_hat is None
    assert "does not exceed the observed count" in est.reason


def test_chapman_interval_is_never_inverted():
    from snowballslr.estimate.capture_recapture import chapman

    for n1, n2, m, obs in ((153, 79, 71, 161), (323, 526, 264, 585), (60, 55, 20, 95)):
        est = chapman(n1, n2, m, n_observed=obs)
        if est.estimable and est.n_hat_ci:
            lo, hi = est.n_hat_ci
            assert lo <= hi, f"inverted interval for n1={n1} n2={n2} m={m}: {lo} > {hi}"


def _capture_table(arms, pattern_counts):
    """Build capture histories from {pattern: count}, pattern aligned with arms."""
    histories, i = {}, 0
    for pattern, n in pattern_counts.items():
        for _ in range(n):
            histories[f"r{i}"] = [a for a, bit in zip(arms, pattern, strict=True) if bit]
            i += 1
    return histories


_LL_ARMS = ("openalex", "crossref", "scopus")


def test_loglinear_refuses_when_an_interaction_cell_is_empty():
    """R2 round 2: the adverse corpus returned n_hat = 1.3e11 instead of refusing.

    openalex is the precedence-first provider, so the crossref-and-scopus-without-
    openalex cell is structurally empty. The all-pairwise model then extrapolates the
    missing cell with no data behind it and the fitted population diverges.
    """
    from snowballslr.estimate.loglinear import loglinear

    counts = {(1, 0, 0): 70, (1, 1, 0): 60, (1, 0, 1): 18, (1, 1, 1): 5,
              (0, 1, 0): 8, (0, 0, 1): 3}
    est = loglinear(_capture_table(_LL_ARMS, counts), _LL_ARMS)
    assert est.estimable is False
    assert est.n_hat is None
    assert "capture cell is empty" in est.reason
    assert ["crossref", "scopus"] in est.detail["empty_pairwise_cells"]


def test_loglinear_refuses_an_arm_that_captured_nothing():
    """A two-arm capture table must not be reported as a three-arm design."""
    from snowballslr.estimate.loglinear import loglinear

    counts = {(1, 0, 0): 82, (1, 1, 0): 71, (0, 1, 0): 8}
    est = loglinear(_capture_table(_LL_ARMS, counts), _LL_ARMS)
    assert est.estimable is False
    assert est.detail["empty_arms"] == ["scopus"]


def test_loglinear_still_estimates_on_a_well_populated_table():
    """The guards must not refuse a capture table that identifies the missing cell."""
    from snowballslr.estimate.loglinear import loglinear

    counts = {(1, 0, 0): 40, (0, 1, 0): 30, (0, 0, 1): 25, (1, 1, 0): 20,
              (1, 0, 1): 18, (0, 1, 1): 15, (1, 1, 1): 12}
    est = loglinear(_capture_table(_LL_ARMS, counts), _LL_ARMS)
    assert est.estimable is True
    assert est.n_hat > est.n_observed


def _recall_history(n_hat, n_obs, trail):
    """Minimal history stub carrying a settled recall estimate."""
    from snowballslr.core.iteration import RecallEstimateRecord
    from snowballslr.estimate.capture_recapture import RecallEstimate

    class _H:
        def __init__(self):
            self.recall_estimate = RecallEstimate(
                method="chapman",
                estimable=True,
                n_observed=n_obs,
                n_hat=n_hat,
                n_hat_ci=(n_hat * 0.98, n_hat * 1.02),
                recall=n_obs / n_hat,
                recall_ci=(n_obs / n_hat, 1.0),
            )
            self.recall_estimate_trail = [
                RecallEstimateRecord(n=k + 1, estimable=True, n_hat=v,
                                     n_observed=n_obs, recall=n_obs / v,
                                     recall_lower=n_obs / v)
                for k, v in enumerate(trail)
            ]

    return _H()


def test_recall_rule_is_advisory_by_default_and_does_not_stop_a_run():
    """R2 revision: the estimate is a diagnostic, not a termination criterion.

    The coverage study shows the estimators are systematically low and their
    nominal intervals badly calibrated, so a reading at or above tau is not
    sufficient evidence to end a review. The rule must still be evaluated and
    recorded -- it simply must not terminate the run.
    """
    from snowballslr.stopping.compose import RuleSet
    from snowballslr.stopping.recall_rule import EstimatedRecall

    rule = EstimatedRecall(tau=0.90)
    assert rule.advisory is True

    hist = _recall_history(100.0, 99, [99.5, 100.0, 100.0])
    decision = rule.evaluate(hist)
    assert decision.triggered is True, "threshold and stability should both be met"
    assert decision.detail["advisory"] is True
    assert "advisory" in decision.rationale

    rs = RuleSet([rule])
    stopped, decisions = rs.evaluate(hist)
    assert stopped is False, "an advisory rule must not terminate the run"
    assert decisions[0] == decision, "the decision must still be recorded"
    assert rs.stopped_by(decisions) is None


def test_recall_rule_can_be_made_authoritative():
    """The opt-in path must still terminate, so the benchmark remains expressible."""
    from snowballslr.stopping.compose import RuleSet
    from snowballslr.stopping.recall_rule import EstimatedRecall

    rule = EstimatedRecall(tau=0.90, authoritative=True)
    assert rule.advisory is False

    hist = _recall_history(100.0, 99, [99.5, 100.0, 100.0])
    decision = rule.evaluate(hist)
    assert decision.triggered is True
    assert decision.detail["advisory"] is False

    rs = RuleSet([rule])
    stopped, decisions = rs.evaluate(hist)
    assert stopped is True
    assert rs.stopped_by(decisions) == "estimated_recall"
