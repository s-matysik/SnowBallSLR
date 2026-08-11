"""Exports, configuration validation and log-linear estimation."""

from __future__ import annotations

import csv

import pytest

from snowballslr.config import Config
from snowballslr.determinism.cache import Cache
from snowballslr.errors import ConfigError, EstimationError
from snowballslr.estimate.loglinear import loglinear
from snowballslr.providers.registry import build_providers, provider_descriptors
from snowballslr.report.export import (
    read_labels,
    write_bibtex,
    write_candidates_csv,
    write_labels_template,
    write_ris,
)
from tests.conftest import make_work


@pytest.fixture
def works():
    return [
        make_work("Alpha study of things", 2020, "Kowalski", "10.1000/a"),
        make_work("Beta analysis", 2019, "Nowak", "10.1000/b"),
    ]


def test_candidates_csv_is_ranked_and_complete(tmp_path, works):
    path = write_candidates_csv(tmp_path / "c.csv", works, {works[1].key: 0.9, works[0].key: 0.1})
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["key"] for r in rows] == [works[1].key, works[0].key]
    assert rows[0]["rank"] == "1" and rows[0]["doi"] == "10.1000/b"


def test_labels_template_roundtrips(tmp_path, works):
    template = write_labels_template(tmp_path / "t.csv", works, {})
    filled = tmp_path / "labels.csv"
    with open(template, newline="", encoding="utf-8") as fh, open(
        filled, "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(["key", "decision", "note"])
        for i, row in enumerate(csv.DictReader(fh)):
            writer.writerow([row["key"], "include" if i == 0 else "", "n"])
    labels = read_labels(filled)
    assert len(labels) == 1 and next(iter(labels.values()))[0] == "include"


def test_ris_export_structure(tmp_path, works):
    text = write_ris(tmp_path / "o.ris", works).read_text(encoding="utf-8")
    assert text.startswith("TY  - ") and "ER  - " in text
    assert text.count("TY  - ") == 2


def test_bibtex_export_disambiguates_duplicate_keys(tmp_path):
    a = make_work("Alpha study of things", 2020, "Kowalski", "10.1000/a")
    b = make_work("Alpha study of things", 2020, "Kowalski", "10.1000/b")
    text = write_bibtex(tmp_path / "o.bib", [a, b]).read_text(encoding="utf-8")
    assert text.count("@article{") == 2
    assert "kowalski2020alpha," in text and "kowalski2020alphab," in text


def test_config_roundtrips_through_yaml(tmp_path):
    cfg = Config.from_dict({"run": {"name": "x", "max_iterations": 3}})
    cfg.to_yaml(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml").run.max_iterations == 3


def test_missing_config_file_raises():
    with pytest.raises(ConfigError):
        Config.from_yaml("/nonexistent/config.yaml")


def test_config_rejects_disabled_provider_in_order():
    cfg = Config.from_dict(
        {"providers": {"order": ["semanticscholar"], "semanticscholar": {"enabled": False}}}
    )
    with pytest.raises(ConfigError):
        cfg.validate_consistency()


def test_config_rejects_forward_with_backward_only_provider():
    cfg = Config.from_dict(
        {
            "run": {"directions": ["forward"]},
            "providers": {"order": ["crossref"], "openalex": {"enabled": False}},
        }
    )
    with pytest.raises(ConfigError):
        cfg.validate_consistency()


def test_config_rejects_embedding_ranker_without_model():
    cfg = Config.from_dict({"ranking": {"type": "embedding"}})
    with pytest.raises(ConfigError):
        cfg.validate_consistency()


def test_config_rejects_empty_directions():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.from_dict({"run": {"directions": []}})


def test_secrets_are_redacted_from_the_hashed_config():
    cfg = Config.from_dict({"providers": {"openalex": {"mailto": "secret@example.org"}}})
    assert "secret@example.org" not in str(cfg.hashable())


def test_provider_registry_builds_in_configured_order(tmp_path):
    cfg = Config.from_dict({"providers": {"order": ["crossref", "openalex"]}})
    providers = build_providers(cfg, Cache(tmp_path / "cache"))
    assert [p.name for p in providers] == ["crossref", "openalex"]
    descriptors = provider_descriptors(providers)
    assert {d["name"] for d in descriptors} == {"crossref", "openalex"}
    assert all("mailto" not in str(d) for d in descriptors)
    for p in providers:
        p.close()


def test_unknown_stopping_rule_is_rejected():
    from snowballslr.stopping.compose import build_rules

    cfg = Config.from_dict({"stopping": {"rules": [{"type": "does_not_exist"}]}})
    with pytest.raises(ConfigError):
        build_rules(cfg.stopping)


def test_loglinear_requires_three_arms():
    with pytest.raises(EstimationError):
        loglinear({"a": ["x", "y"]}, ["x", "y"])


def test_loglinear_estimates_a_missing_cell():
    import random

    rng = random.Random(5)
    histories: dict[str, list[str]] = {}
    for i in range(500):
        arms = [a for a in ("x", "y", "z") if rng.random() < 0.5]
        if arms:
            histories[f"w{i}"] = arms
    est = loglinear(histories, ["x", "y", "z"])
    assert est.estimable
    assert est.n_hat > est.n_observed
    assert "formula" in est.detail
    assert any("model choice" in w for w in est.warnings)


def test_loglinear_handles_empty_input():
    assert not loglinear({}, ["x", "y", "z"]).estimable


PROVIDER_ARMS = {
    "arms": [
        {"name": "openalex", "filter": {"provider": "openalex"}},
        {"name": "crossref", "filter": {"provider": "crossref"}},
    ]
}


def test_legacy_openalex_types_are_flagged():
    """OpenAlex merged journal-article/proceedings-article into `article` in 2023."""
    cfg = Config.from_dict(
        {
            "eligibility": {"types": ["journal-article", "proceedings-article", "book-chapter"]},
            "estimate": PROVIDER_ARMS,
        }
    )
    (warning,) = [w for w in cfg.warnings() if "pre-2023 vocabulary" in w]
    assert "'article'" in warning and "'book-chapter'" in warning


def test_validate_consistency_emits_the_legacy_type_warning():
    from snowballslr.errors import ConfigWarning

    cfg = Config.from_dict({"eligibility": {"types": ["journal-article"]}})
    with pytest.warns(ConfigWarning, match="pre-2023"):
        cfg.validate_consistency()


def test_current_vocabulary_produces_no_warning():
    cfg = Config.from_dict(
        {
            "eligibility": {"types": ["article", "book-chapter", "review"]},
            "estimate": PROVIDER_ARMS,
        }
    )
    assert cfg.warnings() == []


def test_no_warning_when_openalex_is_disabled():
    cfg = Config.from_dict(
        {
            "eligibility": {"types": ["journal-article"]},
            "providers": {"order": ["crossref"], "openalex": {"enabled": False}},
            "estimate": {
                "arms": [
                    {"name": "crossref", "filter": {"provider": "crossref"}},
                    {"name": "grobid", "filter": {"provider": "grobid"}},
                ]
            },
        }
    )
    assert [w for w in cfg.warnings() if "pre-2023" in w] == []


DIRECTION_ARMS = {
    "arms": [
        {"name": "backward", "filter": {"direction": "backward"}},
        {"name": "forward", "filter": {"direction": "forward"}},
    ]
}


def test_direction_arms_are_flagged_as_structurally_degenerate():
    """Direction arms cannot produce an interpretable estimate: on the reference
    run they shared 0 of 478 records."""
    cfg = Config.from_dict({"estimate": DIRECTION_ARMS})
    (warning,) = [w for w in cfg.warnings() if "direction" in w]
    assert "temporally acyclic" in warning and "provider arms" in warning


def test_the_shipped_default_arms_are_not_flagged():
    """The default was changed to provider arms precisely so it warns about nothing."""
    assert Config.from_dict({}).warnings() == []


def test_arms_referencing_disabled_providers_are_flagged():
    cfg = Config.from_dict(
        {
            "providers": {"order": ["crossref"], "openalex": {"enabled": False}},
            "estimate": PROVIDER_ARMS,
        }
    )
    assert any("not enabled in providers.order" in w for w in cfg.warnings())
