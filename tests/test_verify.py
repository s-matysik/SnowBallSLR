"""Drift classification covers every diff class, offline."""

from __future__ import annotations

import copy

from snowballslr.config import canonical_json
from snowballslr.determinism.cache import CacheEntry, sha256_of
from snowballslr.determinism.verify import DiffClass, classify


def _entry(body, method="citations", provider="openalex") -> CacheEntry:
    return CacheEntry(
        cache_key="k",
        provider=provider,
        method=method,
        params={},
        retrieved_at="2026-01-01T00:00:00Z",
        http_status=200,
        response_hash=sha256_of(canonical_json(body)),
        provider_version={},
        body=body,
    )


def test_identical_response_is_unchanged():
    body = {"results": [{"id": "W1"}, {"id": "W2"}]}
    assert classify(_entry(body), copy.deepcopy(body), 200).diff_class == DiffClass.UNCHANGED


def test_added_citing_work_is_new_citing():
    old = {"results": [{"id": "W1"}]}
    new = {"results": [{"id": "W1"}, {"id": "W2"}]}
    diff = classify(_entry(old), new, 200)
    assert diff.diff_class == DiffClass.NEW_CITING
    assert "id:W2" in diff.added


def test_removed_citing_work_is_lost_citing():
    old = {"results": [{"id": "W1"}, {"id": "W2"}]}
    new = {"results": [{"id": "W1"}]}
    diff = classify(_entry(old), new, 200)
    assert diff.diff_class == DiffClass.LOST_CITING
    assert "id:W2" in diff.removed


def test_404_is_deindexed():
    assert classify(_entry({"id": "W1"}), None, 404).diff_class == DiffClass.DEINDEXED


def test_retraction_flag_is_detected():
    old = {"id": "W1", "is_retracted": False}
    new = {"id": "W1", "is_retracted": True}
    assert classify(_entry(old, "resolve"), new, 200).diff_class == DiffClass.RETRACTED


def test_identifier_redirect_is_merged():
    diff = classify(_entry({"id": "W1"}, "resolve"), {"id": "W2"}, 200)
    assert diff.diff_class == DiffClass.MERGED
    assert "W1" in diff.note and "W2" in diff.note


def test_metadata_change_without_membership_change():
    old = {"id": "W1", "title": "Old title", "publication_year": 2020}
    new = {"id": "W1", "title": "New title", "publication_year": 2020}
    diff = classify(_entry(old, "resolve"), new, 200)
    assert diff.diff_class == DiffClass.METADATA_CHANGED
    assert "title" in diff.changed_fields


def test_all_diff_classes_are_reachable():
    """Acceptance criterion 4: injected drift covers every class."""
    seen = {
        classify(_entry({"results": [{"id": "W1"}]}), {"results": [{"id": "W1"}]}, 200).diff_class,
        classify(_entry({"results": [{"id": "W1"}]}), {"results": [{"id": "W1"}, {"id": "W2"}]}, 200).diff_class,
        classify(_entry({"results": [{"id": "W1"}, {"id": "W2"}]}), {"results": [{"id": "W1"}]}, 200).diff_class,
        classify(_entry({"id": "W1", "title": "a"}, "resolve"), {"id": "W1", "title": "b"}, 200).diff_class,
        classify(_entry({"id": "W1", "is_retracted": False}, "resolve"), {"id": "W1", "is_retracted": True}, 200).diff_class,
        classify(_entry({"id": "W1"}, "resolve"), None, 404).diff_class,
        classify(_entry({"id": "W1"}, "resolve"), {"id": "W9"}, 200).diff_class,
    }
    assert seen == set(DiffClass)
