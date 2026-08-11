"""Canonical keys follow the documented precedence and stay stable."""

from __future__ import annotations

from snowballslr.identity.keys import canonical_key, parse_identifier, signature_key


def test_doi_wins_over_all_other_identifiers():
    key = canonical_key(
        doi="https://doi.org/10.1000/ABC",
        source_ids={"openalex": "W123", "s2": "999"},
        title="Whatever",
        year=2020,
    )
    assert key == "doi:10.1000/abc"


def test_openalex_used_when_doi_absent():
    assert canonical_key(source_ids={"openalex": "W123"}, title="T") == "oa:W123"


def test_s2_used_when_doi_and_openalex_absent():
    assert canonical_key(source_ids={"s2": "42"}, title="T") == "s2:42"


def test_signature_fallback_is_stable_across_spellings():
    a = canonical_key(title="Déjà Vu: A Study!", year=2020, first_author_surname="Kowalski")
    b = canonical_key(title="Deja vu - a study", year="2020", first_author_surname="Kowalskí")
    assert a == b == signature_key("Deja vu a study", 2020, "kowalski")


def test_parse_identifier_classifies():
    assert parse_identifier("10.1000/abc") == ("doi", "10.1000/abc")
    assert parse_identifier("https://doi.org/10.1000/abc") == ("doi", "10.1000/abc")
    assert parse_identifier("W2741809807") == ("openalex", "W2741809807")
    assert parse_identifier("12345678")[0] == "pmid"
