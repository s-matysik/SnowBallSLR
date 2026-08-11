"""Normalization is idempotent and stable across spelling variants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from snowballslr.identity.normalize import (
    normalize_doi,
    normalize_surname,
    normalize_title,
    normalize_year,
    tokenize,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/abc", "10.1000/abc"),
        ("http://dx.doi.org/10.1000/abc", "10.1000/abc"),
        ("doi:10.1000/abc", "10.1000/abc"),
        ("  10.1000/abc.  ", "10.1000/abc"),
        ("not-a-doi", None),
        (None, None),
        ("", None),
    ],
)
def test_doi_variants_collapse(raw, expected):
    assert normalize_doi(raw) == expected


def test_title_folds_accents_and_punctuation():
    assert normalize_title("Déjà Vu: A Study!") == normalize_title("Deja vu - a study")


def test_correction_prefix_stripped_even_when_accented():
    assert normalize_title("Erratum: Widget theory") == normalize_title("Widget theory")
    assert normalize_title("Érratum: Widget theory") == normalize_title("Widget theory")


@pytest.mark.parametrize("raw,expected", [("2020", 2020), (2020, 2020), ("2020-05-01", 2020), ("n/a", None), (None, None)])
def test_year_extraction(raw, expected):
    assert normalize_year(raw) == expected


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_normalize_title_idempotent(text):
    once = normalize_title(text)
    assert normalize_title(once) == once


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_normalize_surname_idempotent(text):
    once = normalize_surname(text)
    assert normalize_surname(once) == once


@settings(max_examples=100, deadline=None)
@given(st.text())
def test_normalize_doi_idempotent(text):
    once = normalize_doi(text)
    assert normalize_doi(once) == once


def test_tokenize_drops_stopwords_and_singletons():
    assert tokenize("The a study of X systems") == ["study", "systems"]
