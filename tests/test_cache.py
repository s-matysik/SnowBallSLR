"""Cache is content-addressed, offline-safe and never expires by age."""

from __future__ import annotations

import pytest

from snowballslr.determinism.cache import Cache, cache_key
from snowballslr.errors import OfflineError


def test_cache_key_is_order_insensitive_in_params():
    a = cache_key("openalex", "citations", {"b": 2, "a": 1})
    b = cache_key("openalex", "citations", {"a": 1, "b": 2})
    assert a == b


def test_cache_key_separates_providers_and_methods():
    assert cache_key("openalex", "m", {}) != cache_key("crossref", "m", {})
    assert cache_key("openalex", "a", {}) != cache_key("openalex", "b", {})


def test_second_call_hits_cache_and_does_not_refetch(tmp_path):
    cache = Cache(tmp_path / "cache")
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"value": calls["n"]}, 200, {}

    first = cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, fetch)
    second = cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, fetch)
    assert calls["n"] == 1
    assert first.response_hash == second.response_hash
    assert cache.hits == 1 and cache.misses == 1


def test_refresh_forces_a_refetch(tmp_path):
    cache = Cache(tmp_path / "cache")
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"value": calls["n"]}, 200, {}

    cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, fetch)
    cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, fetch, refresh=True)
    assert calls["n"] == 2


def test_offline_mode_refuses_to_fetch(tmp_path):
    cache = Cache(tmp_path / "cache", offline=True)
    with pytest.raises(OfflineError):
        cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, lambda: ({}, 200, {}))


def test_offline_mode_still_serves_existing_entries(tmp_path):
    warm = Cache(tmp_path / "cache")
    warm.get_or_fetch("openalex", "resolve", {"id": "W1"}, lambda: ({"a": 1}, 200, {}))
    cold = Cache(tmp_path / "cache", offline=True)
    entry = cold.get_or_fetch("openalex", "resolve", {"id": "W1"}, lambda: ({}, 200, {}))
    assert entry.body == {"a": 1}


def test_raw_body_is_stored_unmodified(tmp_path):
    cache = Cache(tmp_path / "cache")
    body = {"nested": {"list": [1, 2, {"x": None}]}, "unicode": "żółć"}
    cache.get_or_fetch("openalex", "resolve", {"id": "W1"}, lambda: (body, 200, {}))
    assert cache.read(cache_key("openalex", "resolve", {"id": "W1"})).body == body
