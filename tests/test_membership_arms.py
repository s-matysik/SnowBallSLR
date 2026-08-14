"""Membership arms: capture arms defined by presence in an external export.

A membership arm is what makes arm independence testable. Two provider arms come
from the same crawl and give a 2x2 table with no residual degree of freedom, so
no model selection is possible; a third arm drawn from a database the crawl did
not use restores it.
"""

from __future__ import annotations

import csv

import pytest

from snowballslr.config import Config
from snowballslr.errors import EstimationError
from snowballslr.estimate import capture_histories, estimate_recall
from snowballslr.estimate.membership import (
    MembershipSet,
    load_membership_sets,
    normalize_doi,
)
from snowballslr.types import Direction, Provenance, Work

ARMS = [
    {"name": "openalex", "filter": {"provider": "openalex"}},
    {"name": "crossref", "filter": {"provider": "crossref"}},
    {"name": "scopus", "filter": {"member_of": "scopus"}},
]


def _work(key: str, doi: str | None, providers: tuple[str, ...]) -> Work:
    prov = tuple(
        Provenance(
            provider=p,
            direction=Direction("backward"),
            parent_key=None,
            iteration=1,
            retrieved_at="2026-01-01T00:00:00Z",
            response_hash="h",
        )
        for p in providers
    )
    return Work(key=key, title=key, title_norm=key, doi=doi, provenance=prov)


def _export(tmp_path, rows, *, column="DOI", name="export.csv"):
    path = tmp_path / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Title", column])
        w.writeheader()
        for i, doi in enumerate(rows):
            w.writerow({"Title": f"paper {i}", column: doi})
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1/AbC", "10.1/abc"),
        ("https://doi.org/10.1/abc", "10.1/abc"),
        ("http://dx.doi.org/10.1/abc", "10.1/abc"),
        ("doi:10.1/ABC", "10.1/abc"),
        ("  10.1/abc.  ", "10.1/abc"),
        ("not-a-doi", None),
        ("", None),
        (None, None),
    ],
)
def test_doi_normalisation_folds_the_forms_exports_actually_use(raw, expected):
    """DOIs are case-insensitive by specification, and exports disagree on prefix."""
    assert normalize_doi(raw) == expected


def test_membership_set_reports_its_own_coverage_ceiling(tmp_path):
    """An export with poor identifier coverage yields an arm that under-captures
    for a mechanical reason. That has to be visible, not inferred."""
    path = _export(tmp_path, ["10.1/a", "10.1/b", "", "not-a-doi"])
    ms = MembershipSet.from_csv("scopus", path)
    assert ms.dois == frozenset({"10.1/a", "10.1/b"})
    assert ms.n_rows == 4
    assert ms.n_without_doi == 2
    assert ms.coverage == 0.5


def test_membership_lookup_is_case_and_prefix_insensitive(tmp_path):
    ms = MembershipSet.from_csv("scopus", _export(tmp_path, ["10.1/AbC"]))
    assert "10.1/abc" in ms
    assert "https://doi.org/10.1/ABC" in ms
    assert "10.1/other" not in ms
    assert None not in ms


def test_missing_doi_column_names_what_it_found(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Title,Identifier\nx,10.1/a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'DOI' column"):
        MembershipSet.from_csv("scopus", path)


def test_long_reference_fields_do_not_break_parsing(tmp_path):
    """Bibliographic exports carry reference lists far beyond csv's default
    field limit; a real Scopus export raised mid-parse before this was handled."""
    path = tmp_path / "long.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["DOI", "References"])
        w.writeheader()
        w.writerow({"DOI": "10.1/a", "References": "; ".join(["ref"] * 40000)})
    assert MembershipSet.from_csv("scopus", path).dois == frozenset({"10.1/a"})
    # The limit must be restored, or it leaks into every later csv read.
    assert csv.field_size_limit() < 2**40


def test_membership_arm_captures_only_records_present_in_the_export(tmp_path):
    ms = MembershipSet.from_csv("scopus", _export(tmp_path, ["10.1/a", "10.1/b"]))
    works = [
        _work("a", "10.1/a", ("openalex", "crossref")),
        _work("b", "10.1/b", ("openalex",)),
        _work("c", "10.9/absent", ("openalex", "crossref")),
    ]
    h = capture_histories(works, ARMS, {"scopus": ms})
    assert h["a"] == ["crossref", "openalex", "scopus"]
    assert h["b"] == ["openalex", "scopus"]
    assert h["c"] == ["crossref", "openalex"]


def test_a_record_without_a_doi_can_never_join_a_membership_arm(tmp_path):
    """Not a defect: it is why MembershipSet reports source coverage. Records the
    crawl could not resolve are exactly the ones a DOI-keyed arm cannot see."""
    ms = MembershipSet.from_csv("scopus", _export(tmp_path, ["10.1/a"]))
    h = capture_histories([_work("d", None, ("crossref",))], ARMS, {"scopus": ms})
    assert h["d"] == ["crossref"]


def test_undeclared_membership_set_raises_rather_than_capturing_nothing(tmp_path):
    """A silently empty arm makes the estimate look unlucky instead of misconfigured."""
    works = [_work("a", "10.1/a", ("openalex",))]
    with pytest.raises(KeyError, match="unknown membership set 'scopus'"):
        capture_histories(works, ARMS, {})


def test_three_arms_make_the_loglinear_estimator_usable(tmp_path):
    """With two arms the 2x2 table has no residual degree of freedom, so model
    selection cannot run. The third arm is what changes that."""
    ms = MembershipSet.from_csv("scopus", _export(tmp_path, ["10.1/a", "10.1/b", "10.1/c"]))
    works = [
        _work("a", "10.1/a", ("openalex", "crossref")),
        _work("b", "10.1/b", ("openalex",)),
        _work("c", "10.1/c", ("crossref",)),
        _work("d", "10.9/absent", ("openalex", "crossref")),
    ]
    est = estimate_recall(
        works, ARMS, method="loglinear", with_diagnostics=False, membership={"scopus": ms}
    )
    assert est.estimable
    assert est.n_hat >= len(works)

    with pytest.raises(EstimationError, match="at least three arms"):
        estimate_recall(works, ARMS[:2], method="loglinear", with_diagnostics=False)


def test_load_membership_sets_keys_by_arm_name(tmp_path):
    a = _export(tmp_path, ["10.1/a"], name="a.csv")
    b = _export(tmp_path, ["10.1/b"], name="b.csv")
    sets = load_membership_sets({"scopus": str(a), "wos": str(b)})
    assert set(sets) == {"scopus", "wos"}
    assert "10.1/a" in sets["scopus"]
    assert "10.1/a" not in sets["wos"]


def test_config_warns_when_an_arm_references_an_undeclared_set():
    cfg = Config.from_dict(
        {"estimate": {"arms": [
            {"name": "openalex", "filter": {"provider": "openalex"}},
            {"name": "scopus", "filter": {"member_of": "scopus"}},
        ]}}
    )
    (warning,) = [w for w in cfg.warnings() if "membership set" in w]
    assert "scopus" in warning and "membership_sets" in warning


def test_config_warns_when_a_declared_export_is_missing(tmp_path):
    cfg = Config.from_dict(
        {"estimate": {
            "arms": [
                {"name": "openalex", "filter": {"provider": "openalex"}},
                {"name": "scopus", "filter": {"member_of": "scopus"}},
            ],
            "membership_sets": {"scopus": str(tmp_path / "absent.csv")},
        }}
    )
    assert any("missing file" in w for w in cfg.warnings())


def test_config_warns_when_a_third_arm_is_configured_but_unused(tmp_path):
    """Chapman uses the first two arms only, so a membership arm added without
    switching method is silently ignored."""
    path = _export(tmp_path, ["10.1/a"])
    cfg = Config.from_dict(
        {"estimate": {
            "method": "chapman",
            "arms": [
                {"name": "openalex", "filter": {"provider": "openalex"}},
                {"name": "crossref", "filter": {"provider": "crossref"}},
                {"name": "scopus", "filter": {"member_of": "scopus"}},
            ],
            "membership_sets": {"scopus": str(path)},
        }}
    )
    (warning,) = [w for w in cfg.warnings() if "only uses the first two" in w]
    assert "loglinear" in warning
