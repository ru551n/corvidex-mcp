"""Tests for the persistent repository state store."""

from __future__ import annotations

import json
from pathlib import Path

from vhdl_rag_mcp.state import RepositoryState, StateStore


def test_empty_store_when_no_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state" / "repositories.json")
    assert store.all() == []
    state = store.get("repo1")
    assert state.name == "repo1"
    assert state.indexed_commit is None
    assert state.last_sync_error is None


def test_set_indexed_persists_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "repositories.json"
    store = StateStore(path)
    store.set_indexed("repo1", "abc123", file_count=42)
    store.record_sync("repo1", None)

    reloaded = StateStore(path)
    state = reloaded.get("repo1")
    assert state.indexed_commit == "abc123"
    assert state.indexed_at is not None
    assert state.last_indexed_file_count == 42
    assert state.last_sync_at is not None
    assert state.last_sync_error is None


def test_failed_sync_records_error_without_touching_commit(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "repositories.json")
    store.set_indexed("repo1", "abc123")
    store.record_sync("repo1", "git fetch failed: connection refused")
    state = store.get("repo1")
    assert state.indexed_commit == "abc123"
    assert state.last_sync_error == "git fetch failed: connection refused"


def test_write_is_valid_json(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "repositories.json")
    store.set_indexed("repo1", "abc123")
    raw = json.loads((tmp_path / "repositories.json").read_text())
    assert raw["repo1"]["indexed_commit"] == "abc123"


def test_corrupt_state_file_is_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "repositories.json"
    path.write_text("{not json", encoding="utf-8")
    store = StateStore(path)  # must not raise
    assert store.all() == []
    assert not path.exists()
    assert (tmp_path / "repositories.corrupt").exists()
    store.set_indexed("repo1", "def456")
    assert store.get("repo1").indexed_commit == "def456"


def test_corrupt_entry_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "repositories.json"
    path.write_text(
        json.dumps({"good": {"name": "good"}, "bad": {"name": 123}}),
        encoding="utf-8",
    )
    store = StateStore(path)
    assert [s.name for s in store.all()] == ["good"]
    assert store.get("good") is not None


def test_datetime_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "repositories.json"
    store = StateStore(path)
    store.set_indexed("repo1", "abc123")
    reloaded = StateStore(path).get("repo1")
    assert reloaded.indexed_at is not None
    assert reloaded.indexed_at.tzinfo is not None


def test_state_model_defaults() -> None:
    state = RepositoryState(name="x")
    assert state.model_dump()["indexed_commit"] is None
