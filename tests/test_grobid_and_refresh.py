"""GROBID reference resolution, drift-refresh reporting, and benchmark driving."""

from __future__ import annotations

import json

from snowballslr.config import Config, canonical_json
from snowballslr.core.run import Run
from snowballslr.determinism.cache import Cache, CacheEntry, sha256_of
from snowballslr.determinism.verify import DiffClass, verify_refresh
from snowballslr.providers.grobid import GrobidProvider, load_pdf_map
from snowballslr.simulation import ReviewSpec, SeedStrategy, run_benchmark
from snowballslr.simulation.oracle import Oracle, load_gold_standard
from snowballslr.types import Work
from tests.conftest import make_work

TEI_REFS = [
    {"title": "Guidelines for snowballing", "authors": ["Wohlin"], "year": 2014,
     "venue": "EASE", "doi": None},
    {"title": "A study that will not match", "authors": ["Nobody"], "year": 1999,
     "venue": None, "doi": None},
    {"title": "Already has an identifier", "authors": ["Smith"], "year": 2020,
     "venue": None, "doi": "10.1000/known"},
]


class FakeCrossref:
    """Stands in for the Crossref matcher so no network is required."""

    def __init__(self, matches):
        self.matches = matches
        self.calls = 0

    def match_bibliographic(self, query, rows=3):
        self.calls += 1
        return self.matches


def _grobid(tmp_path, crossref, monkeypatch):
    provider = GrobidProvider(
        Cache(tmp_path / "cache"),
        pdf_map={"doi:10.1000/parent": str(tmp_path / "paper.pdf")},
        crossref=crossref,
    )
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(provider, "_process_pdf", lambda key, path: TEI_REFS)
    return provider


def test_grobid_resolves_matching_references_and_flags_the_rest(tmp_path, monkeypatch):
    target = make_work("Guidelines for snowballing", 2014, "Wohlin", "10.1145/2601248.2601268")
    crossref = FakeCrossref([(target, 95.0)])
    provider = _grobid(tmp_path, crossref, monkeypatch)

    parent = Work(key="doi:10.1000/parent", title="Parent", title_norm="parent")
    refs = provider.references(parent)

    by_title = {r.title: r for r in refs}
    assert by_title["Guidelines for snowballing"].doi == "10.1145/2601248.2601268"
    assert by_title["A study that will not match"].unresolved
    assert by_title["Already has an identifier"].doi == "10.1000/known"
    assert provider.match_stats["matched"] == 1
    assert provider.match_stats["unresolved"] == 1


def test_grobid_rejects_low_scoring_matches(tmp_path, monkeypatch):
    target = make_work("Guidelines for snowballing", 2014, "Wohlin", "10.1145/2601248.2601268")
    provider = _grobid(tmp_path, FakeCrossref([(target, 10.0)]), monkeypatch)
    refs = provider.references(Work(key="doi:10.1000/parent", title="P", title_norm="p"))
    assert all(r.unresolved for r in refs if r.title != "Already has an identifier")


def test_grobid_rejects_year_mismatch(tmp_path, monkeypatch):
    wrong_year = make_work("Guidelines for snowballing", 1990, "Wohlin", "10.1145/2601248.2601268")
    provider = _grobid(tmp_path, FakeCrossref([(wrong_year, 95.0)]), monkeypatch)
    refs = provider.references(Work(key="doi:10.1000/parent", title="P", title_norm="p"))
    assert next(r for r in refs if r.title == "Guidelines for snowballing").unresolved


def test_grobid_is_backward_only_and_skips_unknown_pdfs(tmp_path, monkeypatch):
    provider = _grobid(tmp_path, FakeCrossref([]), monkeypatch)
    unknown = Work(key="doi:10.1000/other", title="X", title_norm="x")
    assert provider.references(unknown) == []
    assert provider.citations(unknown) == []
    assert provider.resolve("anything") is None


def test_pdf_map_loading(tmp_path):
    p = tmp_path / "pdfs.csv"
    p.write_text("key,path\ndoi:10.1000/a,/tmp/a.pdf\n,ignored\n", encoding="utf-8")
    assert load_pdf_map(p) == {"doi:10.1000/a": "/tmp/a.pdf"}


# -- refresh -------------------------------------------------------------


def _seed_cache(cache: Cache, bodies: dict[str, dict]) -> None:
    for key, body in bodies.items():
        cache.write(
            CacheEntry(
                cache_key=key,
                provider="openalex",
                method="citations" if key.startswith("c") else "resolve",
                params={},
                retrieved_at="2026-01-01T00:00:00Z",
                http_status=200,
                response_hash=sha256_of(canonical_json(body)),
                provider_version={},
                body=body,
            )
        )


def test_refresh_classifies_and_reports_every_drift_class(tmp_path, graph, seeds, config, oracle):
    root = tmp_path / "r"
    run = Run.init(root, seeds, config, offline_graph=graph)
    run.run_to_saturation(oracle.as_mapping())
    run.report()

    cache = Cache(root / "cache")
    _seed_cache(
        cache,
        {
            "c" + "0" * 63: {"results": [{"id": "W1"}]},
            "r" + "1" * 63: {"id": "W2", "is_retracted": False, "title": "t"},
            "r" + "2" * 63: {"id": "W3", "title": "old"},
            "r" + "3" * 63: {"id": "W4"},
        },
    )

    def fetcher(entry):
        body = entry.body
        if entry.cache_key.startswith("c"):
            return {"results": [{"id": "W1"}, {"id": "W99"}]}, 200
        if body.get("id") == "W2":
            return {**body, "is_retracted": True}, 200
        if body.get("id") == "W3":
            return {**body, "title": "new"}, 200
        return None, 404

    report = verify_refresh(root, fetcher)
    assert report.counts.get(DiffClass.NEW_CITING.value, 0) >= 1
    assert report.counts[DiffClass.RETRACTED.value] == 1
    assert report.counts[DiffClass.METADATA_CHANGED.value] == 1
    assert report.counts[DiffClass.DEINDEXED.value] == 1
    assert not report.ok  # deindexed records make the run non-reproducible as-is

    json_path, md_path = report.write(root)
    text = md_path.read_text(encoding="utf-8")
    assert "Drift classification" in text and "retracted" in text
    assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == "refresh"
    assert (root / "cache_refresh").exists()  # original cache untouched
    run.close()


def test_refresh_limit_caps_work(tmp_path, graph, seeds, config, oracle):
    root = tmp_path / "r"
    run = Run.init(root, seeds, config, offline_graph=graph)
    run.run_to_saturation(oracle.as_mapping())
    run.report()
    _seed_cache(Cache(root / "cache"), {f"{i}" * 64: {"id": f"W{i}"} for i in range(5)})
    report = verify_refresh(root, lambda e: (e.body, 200), limit=2)
    assert report.entries_checked == 2
    run.close()


# -- oracle and benchmark -------------------------------------------------


def test_oracle_loads_from_json_and_from_doi_columns(tmp_path):
    j = tmp_path / "gold.json"
    j.write_text(json.dumps([{"doi": "10.1000/a"}, {"key": "doi:10.1000/b"}]), encoding="utf-8")
    assert load_gold_standard(j) == {"doi:10.1000/a", "doi:10.1000/b"}

    c = tmp_path / "gold.csv"
    c.write_text("doi,title\n10.1000/c,\n,Some title here\n", encoding="utf-8")
    keys = load_gold_standard(c)
    assert "doi:10.1000/c" in keys and len(keys) == 2


def test_oracle_decide_and_membership():
    o = Oracle.from_keys(["a", "b"])
    assert o.decide("a") == "include" and o.decide("z") == "exclude"
    assert "a" in o and len(o) == 2
    assert o.as_mapping(["a", "z"]) == {"a": "include", "z": "exclude"}


def test_benchmark_runs_a_small_factorial_design(tmp_path, graph, oracle):
    spec = ReviewSpec(name="mini", graph=graph, gold=oracle.included, domain="test")
    summary = run_benchmark(
        [spec],
        tmp_path,
        strategies=[SeedStrategy.MOST_CITED, SeedStrategy.RANDOM],
        seed_sizes=[2],
        repeats=2,
        config=Config.from_dict({"run": {"max_iterations": 5}}),
        out=tmp_path / "results.json",
    )
    assert summary["n_trials"] == 3  # 1 deterministic + 2 random repeats
    assert summary["mean_final_recall"] > 0.5
    assert (tmp_path / "results.json").exists()
    assert "calibration" in summary


def test_review_spec_loads_from_files(tmp_path, graph):
    g = tmp_path / "graph.json"
    g.write_text(json.dumps(graph), encoding="utf-8")
    gold = tmp_path / "gold.csv"
    gold.write_text("key\ndoi:10.1000/gold000\n", encoding="utf-8")
    spec = ReviewSpec.from_files("x", g, gold, domain="d", boolean_seeds=["doi:10.1000/gold000"])
    assert spec.domain == "d" and len(spec.gold) == 1
