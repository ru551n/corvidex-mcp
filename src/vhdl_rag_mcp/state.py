"""Persistent per-repository indexing state.

A small JSON document (``state/repositories.json``) tracks the last
successfully indexed commit and synchronization metadata per
repository. The document carries an explicit ``schema_version`` (the
index layout version); legacy v1 documents (a flat repository map,
predating the hdl collection) are detected on load and migrated by
:meth:`StateStore.migrate` — a deterministic full reindex.
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

from .models import INDEX_SCHEMA_VERSION

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
    #: Local working and filesystem repositories only: cheap fingerprint
    #: (HEAD + porcelain status, or the filesystem walk) of the working
    #: tree at the last successful sync. The fast local poller compares
    #: its freshly computed fingerprint to this value to decide whether
    #: a sync is needed without running the full plan.
    local_fingerprint: str | None = None
    #: Local working and filesystem repositories only: content
    #: fingerprints (sha256) of untracked / walked files at the last
    #: successful sync, keyed by repository-relative path. Lets the sync
    #: plan skip re-chunking unchanged files and detect deleted ones.
    untracked_indexed: dict[str, str] = {}
    #: Gitlink (submodule) path -> submodule SHA last fully indexed.
    #: Lets an incremental sync diff each submodule against its
    #: previously indexed content.
    submodules: dict[str, str] = {}


class StateStore:
    """Loads/saves repository state with atomic writes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._states: dict[str, RepositoryState] = {}
        # Index layout version of the persisted document (a fresh
        # store starts at the current version).
        self._schema_version = INDEX_SCHEMA_VERSION
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt state file must not kill the server; the index in
            # the vector store remains usable. Log loudly and start empty — the next
            # sync reindexes conservatively (full index when commit is
            # unknown).
            logger.error(
                "state file %s is unreadable (%s); starting with empty state",
                self._path,
                exc,
            )
            self._path.rename(self._path.with_suffix(".corrupt"))
            return
        if "schema_version" in raw:
            version = raw.get("schema_version")
            repositories: Any = raw.get("repositories") or {}
        else:
            # v1 layout: a flat name -> state document (pre-hdl).
            version = 1
            repositories = raw
        if isinstance(version, int):
            self._schema_version = version
        if self._schema_version > INDEX_SCHEMA_VERSION:
            logger.warning(
                "state file %s was written by a newer schema (v%d); "
                "keeping the newer document as-is",
                self._path,
                self._schema_version,
            )
        if not isinstance(repositories, dict):
            repositories = {}
        for name, data in repositories.items():
            try:
                self._states[name] = RepositoryState.model_validate(data)
            except Exception as exc:
                logger.error("skipping corrupt state entry %r: %s", name, exc)

    def save(self) -> None:
        """Atomically persist all state in the current schema layout."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "repositories": {
                name: state.model_dump(mode="json")
                for name, state in self._states.items()
            },
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

    @property
    def schema_version(self) -> int:
        """Index layout version of the loaded (or fresh) document."""
        return self._schema_version

    @property
    def needs_migration(self) -> bool:
        """True when the persisted layout predates the current schema."""
        return self._schema_version < INDEX_SCHEMA_VERSION

    def reset_all_indexed(self, save: bool = True) -> None:
        """Forget every repository's indexed commit: the next sync
        reindexes each one fully and deterministically."""
        for state in self._states.values():
            state.indexed_commit = None
            state.indexed_at = None
        if save:
            self.save()

    def migrate(self) -> bool:
        """Migrate a legacy (v1) document to the current schema.

        A v1 document predates the hdl collection layout, so the
        indexed commits it tracks are invalid and are forgotten; the
        next sync rebuilds the index deterministically from git.
        Returns True when a migration ran; idempotent on a current
        document.
        """
        if not self.needs_migration:
            return False
        self.reset_all_indexed(save=False)
        self._schema_version = INDEX_SCHEMA_VERSION
        self.save()
        return True

    def set_indexed(
        self,
        name: str,
        commit: str,
        file_count: int = 0,
        save: bool = True,
        submodules: dict[str, str] | None = None,
    ) -> None:
        """Mark a commit as fully indexed. Call only after the index update
        (embeddings, upserts, deletions) has succeeded. ``submodules``
        replaces the persisted per-gitlink SHA map when given; ``None``
        leaves the stored map untouched."""
        state = self.get(name)
        state.indexed_commit = commit
        state.indexed_at = _utcnow()
        state.last_indexed_file_count = file_count
        if submodules is not None:
            state.submodules = dict(submodules)
        if save:
            self.save()

    def record_sync(self, name: str, error: str | None, save: bool = True) -> None:
        state = self.get(name)
        state.last_sync_at = _utcnow()
        state.last_sync_error = error
        if save:
            self.save()
