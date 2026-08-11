"""Regression tests for defects found during the SoftwareX validation study.

Each test names the failure it locks down. All are offline and deterministic.
"""

from __future__ import annotations

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
