"""Persistent per-repository indexing state.

A small JSON document (``state/repositories.json``) tracks the last
successfully indexed commit and synchronization metadata per repository.
Writes are atomic (temp file + ``os.replace``) so a crash never corrupts
state. Only commit state that reflects a fully successful index update is
persisted; a failed index run leaves the previous state untouched.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RepositoryState(BaseModel):
    """Indexing/sync state for one repository."""

    name: str
    #: Last commit fully indexed (chunks embedded + upserted). None until
    #: the first successful index run.
    indexed_commit: str | None = None
    indexed_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_sync_error: str | None = None
    #: Last full index run (clone or reindex).
    last_indexed_file_count: int = 0


class StateStore:
    """Loads/saves repository state with atomic writes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._states: dict[str, RepositoryState] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt state file must not kill the server; the index in
            # Qdrant remains usable. Log loudly and start empty — the next
            # sync reindexes conservatively (full index when commit is
            # unknown).
            logger.error(
                "state file %s is unreadable (%s); starting with empty state",
                self._path,
                exc,
            )
            self._path.rename(self._path.with_suffix(".corrupt"))
            return
        for name, data in raw.items():
            try:
                self._states[name] = RepositoryState.model_validate(data)
            except Exception as exc:
                logger.error("skipping corrupt state entry %r: %s", name, exc)

    def save(self) -> None:
        """Atomically persist all state."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: state.model_dump(mode="json") for name, state in self._states.items()
        }
        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def get(self, name: str) -> RepositoryState:
        if name not in self._states:
            self._states[name] = RepositoryState(name=name)
        return self._states[name]

    def all(self) -> list[RepositoryState]:
        return [self._states[name] for name in sorted(self._states)]

    def remove(self, name: str) -> None:
        """Forget one repository's state (config removal)."""
        if name in self._states:
            del self._states[name]
            self.save()

    def set_indexed(
        self, name: str, commit: str, file_count: int = 0, save: bool = True
    ) -> None:
        """Mark a commit as fully indexed. Call only after the index update
        (embeddings, upserts, deletions) has succeeded."""
        state = self.get(name)
        state.indexed_commit = commit
        state.indexed_at = _utcnow()
        state.last_indexed_file_count = file_count
        if save:
            self.save()

    def record_sync(self, name: str, error: str | None, save: bool = True) -> None:
        state = self.get(name)
        state.last_sync_at = _utcnow()
        state.last_sync_error = error
        if save:
            self.save()
