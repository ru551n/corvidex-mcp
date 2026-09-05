"""Tests for the persistent repository state store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from corvidex_mcp.models import INDEX_SCHEMA_VERSION
from corvidex_mcp.state import STATE_SCHEMA_VERSION, RepositoryState, StateStore

INDEX = "index.sqlite"
LEGACY = "repositories.json"


def make_store(tmp_path: Path, legacy: Path | None = None) -> StateStore:
    return StateStore(tmp_path / INDEX, legacy)


def test_empty_store_when_no_file(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.all() == []
    state = store.get("repo1")
    assert state.name == "repo1"
    assert state.indexed_commit is None
    assert state.last_sync_error is None
    assert state.local_fingerprint is None
    assert state.untracked_indexed == {}
    store.close()


def test_set_indexed_persists_across_reload(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123", file_count=42)
    store.record_sync("repo1", None)
    store.close()

    reloaded = make_store(tmp_path)
    state = reloaded.get("repo1")
    assert state.indexed_commit == "abc123"
    assert state.indexed_at is not None
    assert state.last_indexed_file_count == 42
    assert state.last_sync_at is not None
    assert state.last_sync_error is None
    reloaded.close()


def test_failed_sync_records_error_without_touching_commit(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    store.record_sync("repo1", "git fetch failed: connection refused")
    state = store.get("repo1")
    assert state.indexed_commit == "abc123"
    assert state.last_sync_error == "git fetch failed: connection refused"
    store.close()


def test_state_lives_in_the_index_database(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    conn = sqlite3.connect(tmp_path / INDEX)
    try:
        row = conn.execute(
            "SELECT indexed_commit FROM repositories WHERE name = ?", ("repo1",)
        ).fetchone()
    finally:
        conn.close()
    assert row == ("abc123",)
    store.close()


def test_state_schema_is_stamped_in_meta(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.schema_version == INDEX_SCHEMA_VERSION
    conn = sqlite3.connect(tmp_path / INDEX)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'state_schema_version'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None and int(row[0]) == STATE_SCHEMA_VERSION
    store.close()


def test_corrupt_legacy_file_is_quarantined(tmp_path: Path) -> None:
    legacy = tmp_path / LEGACY
    legacy.write_text("{not json", encoding="utf-8")
    store = make_store(tmp_path, legacy)  # must not raise
    assert store.all() == []
    assert not store.needs_migration
    assert not legacy.exists()
    assert (tmp_path / "repositories.corrupt").exists()
    store.set_indexed("repo1", "def456")
    assert store.get("repo1").indexed_commit == "def456"
    store.close()


def test_corrupt_legacy_entry_is_skipped(tmp_path: Path) -> None:
    legacy = tmp_path / LEGACY
    legacy.write_text(
        json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "repositories": {"good": {"name": "good"}, "bad": {"name": 123}},
            }
        ),
        encoding="utf-8",
    )
    store = make_store(tmp_path, legacy)
    assert store.migrate() is True
    assert [s.name for s in store.all()] == ["good"]
    store.close()


def test_datetime_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    store.close()
    reloaded = make_store(tmp_path).get("repo1")
    assert reloaded.indexed_at is not None
    assert reloaded.indexed_at.tzinfo is not None


def test_untracked_fingerprints_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    state = store.get("local1")
    state.local_fingerprint = "fp-123"
    state.untracked_indexed = {"a.vhd": "sha-a", "b.vhd": "sha-b"}
    store.record_sync("local1", None)
    store.close()

    reloaded = make_store(tmp_path).get("local1")
    assert reloaded.local_fingerprint == "fp-123"
    assert reloaded.untracked_indexed == {"a.vhd": "sha-a", "b.vhd": "sha-b"}


def test_state_model_defaults() -> None:
    state = RepositoryState(name="x")
    assert state.model_dump()["indexed_commit"] is None


def test_v1_legacy_document_is_detected_and_forgotten(tmp_path: Path) -> None:
    legacy = tmp_path / LEGACY
    legacy.write_text(
        json.dumps({"repo1": {"name": "repo1", "indexed_commit": "abc123"}}),
        encoding="utf-8",
    )
    store = make_store(tmp_path, legacy)
    assert store.needs_migration
    assert store.migrate() is True
    # v1 documents predate the current index layout: commits forgotten.
    assert store.get("repo1").indexed_commit is None
    assert not store.needs_migration
    # The import runs exactly once.
    assert not legacy.exists()
    assert (tmp_path / "repositories.json.migrated").exists()
    assert store.migrate() is False
    store.close()


def test_v2_legacy_document_is_imported_as_is(tmp_path: Path) -> None:
    legacy = tmp_path / LEGACY
    legacy.write_text(
        json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "repositories": {
                    "repo1": {
                        "name": "repo1",
                        "indexed_commit": "abc123",
                        "last_indexed_file_count": 7,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = make_store(tmp_path, legacy)
    assert store.migrate() is True
    state = store.get("repo1")
    assert state.indexed_commit == "abc123"
    assert state.last_indexed_file_count == 7
    store.close()


def test_legacy_import_never_overwrites_existing_rows(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "existing-commit")
    store.close()

    legacy = tmp_path / LEGACY
    legacy.write_text(
        json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "repositories": {
                    "repo1": {"name": "repo1", "indexed_commit": "other-commit"},
                    "repo2": {"name": "repo2"},
                },
            }
        ),
        encoding="utf-8",
    )
    reloaded = make_store(tmp_path, legacy)
    assert reloaded.migrate() is True
    assert reloaded.get("repo1").indexed_commit == "existing-commit"
    assert reloaded.get("repo2").name == "repo2"
    reloaded.close()


def test_migrate_is_idempotent_on_current_deployment(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    assert store.migrate() is False
    store.close()

    reloaded = make_store(tmp_path)
    assert reloaded.migrate() is False
    assert reloaded.schema_version == INDEX_SCHEMA_VERSION
    assert not reloaded.needs_migration
    assert reloaded.get("repo1").indexed_commit == "abc123"
    reloaded.close()


def test_reset_all_indexed_forgets_commits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    store.set_indexed("repo2", "def456")
    store.reset_all_indexed()
    assert store.get("repo1").indexed_commit is None
    assert store.get("repo2").indexed_commit is None
    store.close()

    reloaded = make_store(tmp_path)
    assert reloaded.get("repo1").indexed_commit is None
    assert reloaded.get("repo2").indexed_commit is None
    reloaded.close()


def test_remove_forgets_state(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_indexed("repo1", "abc123")
    store.remove("repo1")
    assert store.all() == []
    store.close()

    reloaded = make_store(tmp_path)
    assert reloaded.all() == []
    reloaded.close()
