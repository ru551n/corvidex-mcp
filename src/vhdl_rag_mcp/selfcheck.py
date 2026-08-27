"""Startup self-check: verify runtime components before serving.

The server is a single local tool, so failures should surface at
startup with an actionable message rather than mid-sync. Two classes:

* **required** — the server cannot serve without these; a failure
  aborts startup with a clear error:
    - ``git`` — repositories are cloned and synced via the git binary;
    - ``sqlite`` — the embedded index runtime (stdlib SQLite version);
    - ``fts5`` — the full-text leg of hybrid search (the FTS5 module
      must be compiled into the SQLite build);
    - ``sqlite-vec`` — the vector leg of hybrid search (loadable
      extension; probed by loading it into a scratch connection and
      reading ``vec_version()``);
    - ``schema`` — the index layout is at the current version after
      migration.
* **degraded** — the server keeps operating with reduced capability
  and reports what is missing:
    - the dense embedding models (one per collection): a model that
      fails to load — e.g. not present in the offline model cache —
      leaves that collection's embedding-based search and indexing
      unavailable until the model is provisioned; lexical search is
      unaffected;
    - the HDL analyzers (vhdl_ls, Veridian): their files fall back to
      structural parsing (the existing graceful fallback).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import AppConfig
from .lsp.analyzers import build_analyzer_statuses
from .models import INDEX_SCHEMA_VERSION
from .vector_store import ALL_COLLECTIONS, VectorStore

if TYPE_CHECKING:
    from .server import VhdlRagApp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentStatus:
    """One component check result.

    ``ok`` is False for a broken component; ``optional`` marks the
    degradable ones (a broken optional component degrades the server,
    a broken required one aborts startup).
    """

    name: str
    ok: bool
    optional: bool
    detail: str


@dataclass(frozen=True)
class SelfCheckResult:
    """All component checks, in report order."""

    components: tuple[ComponentStatus, ...]

    @property
    def required_ok(self) -> bool:
        """True when every required component is ok."""
        return all(c.ok for c in self.components if not c.optional)

    @property
    def degraded(self) -> tuple[str, ...]:
        """Names of broken components (required or optional)."""
        return tuple(c.name for c in self.components if not c.ok)

    def summary(self) -> str:
        if not self.degraded:
            return "ok"
        required = tuple(c.name for c in self.components if not c.ok and not c.optional)
        optional = tuple(c.name for c in self.components if not c.ok and c.optional)
        parts = []
        if required:
            parts.append("FATAL: " + ", ".join(required))
        if optional:
            parts.append("degraded: " + ", ".join(optional))
        return "; ".join(parts)


# -- required components ------------------------------------------------------


def check_git() -> ComponentStatus:
    path = shutil.which("git")
    if path is None:
        return ComponentStatus("git", False, False, "git not found on PATH")
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return ComponentStatus("git", True, False, f"{out} ({path})")
    except (OSError, subprocess.SubprocessError) as exc:
        return ComponentStatus("git", False, False, f"git --version failed: {exc}")


def check_sqlite() -> ComponentStatus:
    # The stdlib module imported by the process IS the runtime the index
    # uses; reporting its version doubles as the presence check.
    return ComponentStatus("sqlite", True, False, sqlite3.sqlite_version)


def check_fts5() -> ComponentStatus:
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE vrmcp_selfcheck USING fts5(x)")
            conn.execute("DROP TABLE vrmcp_selfcheck")
        finally:
            conn.close()
        return ComponentStatus("fts5", True, False, "available")
    except sqlite3.Error as exc:
        return ComponentStatus("fts5", False, False, f"not available: {exc}")


def check_sqlite_vec() -> ComponentStatus:
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            version = conn.execute("SELECT vec_version()").fetchone()[0]
        finally:
            conn.close()
        return ComponentStatus("sqlite-vec", True, False, str(version))
    except Exception as exc:
        return ComponentStatus("sqlite-vec", False, False, f"load failed: {exc}")


def check_schema(store: VectorStore) -> ComponentStatus:
    version = store.schema_version
    if version == INDEX_SCHEMA_VERSION:
        return ComponentStatus("schema", True, False, f"v{version}")
    return ComponentStatus(
        "schema",
        False,
        False,
        f"v{version} != current v{INDEX_SCHEMA_VERSION}",
    )


# -- degraded components ------------------------------------------------------


def check_models(app: VhdlRagApp) -> list[ComponentStatus]:
    """One entry per collection from the app's recorded model errors
    (populated by ``VhdlRagApp.ensure_collections``)."""
    statuses: list[ComponentStatus] = []
    for collection in ALL_COLLECTIONS:
        error = app.collection_error(collection)
        if error is None:
            statuses.append(
                ComponentStatus(
                    f"model:{collection.value}",
                    True,
                    True,
                    app.providers.model_name(collection),
                )
            )
        else:
            statuses.append(
                ComponentStatus(f"model:{collection.value}", False, True, error)
            )
    return statuses


def check_analyzers(config: AppConfig) -> list[ComponentStatus]:
    statuses: list[ComponentStatus] = []
    for analyzer in build_analyzer_statuses(
        config.vhdl_ls_path, config.veridian_path
    ).values():
        if analyzer.available:
            detail = f"{analyzer.mode}, {analyzer.version}"
            if analyzer.path:
                detail += f" ({analyzer.path})"
            statuses.append(ComponentStatus(analyzer.name, True, True, detail))
        else:
            statuses.append(
                ComponentStatus(
                    analyzer.name, False, True, analyzer.error or "unavailable"
                )
            )
    return statuses


def run_self_check(app: VhdlRagApp) -> SelfCheckResult:
    """Run every component check (the schema check reads the store after
    migration; the model checks read the errors recorded while the
    collections were being ensured)."""
    return SelfCheckResult(
        (
            check_git(),
            check_sqlite(),
            check_fts5(),
            check_sqlite_vec(),
            check_schema(app.store),
            *check_models(app),
            *check_analyzers(app.config),
        )
    )
