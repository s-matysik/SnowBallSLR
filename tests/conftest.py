"""Shared fixtures. INV-4: the suite must never touch the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snowballslr import Config
from snowballslr.identity.keys import canonical_key
from snowballslr.identity.normalize import normalize_doi, normalize_title
from snowballslr.simulation.oracle import Oracle
from snowballslr.types import Author, Work

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Hard-fail any attempted HTTP call from within the test suite."""
    import httpx

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in tests (INV-4 violated)")

    monkeypatch.setattr(httpx.Client, "get", _blocked)
    monkeypatch.setattr(httpx.Client, "post", _blocked)
    monkeypatch.setattr(httpx, "get", _blocked)
    monkeypatch.setattr(httpx, "post", _blocked)


@pytest.fixture(scope="session")
def graph() -> dict:
    return json.loads((FIXTURES / "mini_graph.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def oracle() -> Oracle:
    return Oracle.from_file(FIXTURES / "mini_gold.csv")


@pytest.fixture
def seeds(oracle: Oracle) -> list[str]:
    return sorted(oracle.included)[:3]


@pytest.fixture
def config() -> Config:
    return Config.from_dict(
        {
            "run": {"name": "test", "max_iterations": 8},
            "stopping": {
                "mode": "any_of",
                "rules": [
                    {"type": "marginal_yield", "eps": 0.01, "k": 2},
                    {"type": "budget", "max_screened": 5000},
                    {"type": "exhaustion"},
                ],
            },
        }
    )


def make_work(title: str, year: int | None, surname: str | None, doi: str | None = None) -> Work:
    key = canonical_key(doi=doi, title=title, year=year, first_author_surname=surname)
    return Work(
        key=key,
        title=title,
        title_norm=normalize_title(title),
        doi=normalize_doi(doi),
        year=year,
        authors=(Author(surname=surname),) if surname else (),
    )


@pytest.fixture
def mk():
    return make_work
