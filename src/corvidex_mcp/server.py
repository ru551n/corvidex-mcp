"""MCP server: RTL-centric RAG and indexing over configured Git repositories.

Exposes ten tools to coding agents:

- ``search_hdl`` / ``search_vhdl`` / ``search_docs`` / ``search_code`` —
  hybrid
  (dense + full-text) semantic search in one domain, with optional
  repository/category filters and identifier cross-references;
- ``search_knowledge`` — the same search fused across all three
  domains (RRF over the per-domain rank lists);
- ``get_source`` — exact file content (or a line range) from the
  synced working tree, with repository/commit attribution;
- ``repository_files`` — list the indexed files of a repository
  (glob-filterable), i.e. the candidate paths for ``get_source``;
- ``repository_status`` — what is indexed and any sync errors;
- ``sync_repositories`` / ``reindex_repository`` — maintenance
  (incremental sync of selected repos, full reindex of one).

Lifecycle: load config, log to stderr/file (stdout is reserved for
the MCP protocol), take a single-instance lock, create the vector-store
tables (loading the embedding models from the local model cache), run
the startup self-check (required components abort startup; optional
ones degrade), migrate the index, run an initial sync, then serve
stdio while a background task syncs every ``sync_interval`` seconds.
All failures are contained per repository:
one broken repository records its error and does not affect the
others or stop the server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ModuleNotFoundError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from .config import (
    CODING_STANDARDS_REPO,
    AppConfig,
    ConfigError,
    RepositoryConfig,
    apply_default_repository,
    load_config,
)
from .embeddings.providers import EmbeddingProviders
from .git_manager import GitManager
from .indexing import IndexPipeline
from .logging_setup import setup_logging
from .lsp import AnalyzerStatus, build_analyzer_statuses
from .models import INDEX_SCHEMA_VERSION, CollectionName, SearchResult
from .retrieval import RetrievalError, RetrievalService
from .selfcheck import SelfCheckResult, run_self_check
from .standards import sync_coding_standards
from .state import StateStore
from .vector_store import ALL_COLLECTIONS, VectorStore

logger = logging.getLogger(__name__)

#: Held open for the process lifetime while the lock is taken.
_LOCK_HANDLE: object | None = None

MCP_NAME = "corvidex_mcp"

INSTRUCTIONS = (
    "Semantic search over an organization's HDL code (VHDL, Verilog, "
    "SystemVerilog), HDL-related documentation, and general source code "
    "(C/C++, Python, ...). Use search_hdl for reference HDL "
    "implementations (entities/modules, architectures, processes/always "
    "blocks, packages, functions, tasks, reset/clock/FSM patterns) with "
    "an optional language filter ('vhdl' | 'verilog' | "
    "'systemverilog'); search_vhdl is the VHDL-only form of search_hdl. "
    "search_docs for standards and design documentation, search_code for "
    "general C/C++/Python code, and search_knowledge when the answer may "
    "span any of them. Pass `symbols` to find every chunk that references "
    "specific identifiers (cross-referencing; works across HDL "
    "languages). Every search tool takes a `mode` strategy: 'hybrid' "
    "(default; semantic + full-text, RRF-fused), 'semantic' (embedding "
    "similarity only), or 'lexical' (full-text match only). Use "
    "get_source for the full text of a known file (exact "
    "lines, exact commit). Every result carries source attribution "
    "(repository, file, line range, commit, language). Use "
    "repository_status to see what is indexed and whether a sync failed; "
    "it also reports the HDL analyzer status (vhdl_ls / Veridian). "
    "sync_repositories to force an update, reindex_repository to rebuild "
    "one repository's index. When a coding-standards file is "
    "configured it is indexed as the 'coding-standards' "
    "pseudo-repository with a high retrieval priority: search with "
    "repository='coding-standards' to restrict to it. A search result "
    "may start with a 'Note: ... still indexing' line when a repository "
    "has not finished its initial sync yet (or is being resynced): "
    "results may be thin or empty in that case — wait a few seconds and "
    "retry rather than concluding nothing exists."
)

_READ_ONLY = ToolAnnotations(read_only_hint=True)
_READ_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)

DEFAULT_LIMIT = 8
KNOWLEDGE_LIMIT = 10


def _local_poll_done(task: asyncio.Task[None], name: str, in_flight: set[str]) -> None:
    """Clear the in-flight flag and log a failed local poll sync.

    The fast poller fires repository syncs as detached tasks so a slow
    sync never blocks the polling loop; this is their completion
    callback.
    """
    in_flight.discard(name)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("%s: local poll sync failed: %s", name, exc)


class VhdlRagApp:
    """Runtime components and lifecycle for one server process."""

    def __init__(
        self,
        config: AppConfig,
        providers: EmbeddingProviders | None = None,
        store: VectorStore | None = None,
        states: StateStore | None = None,
    ) -> None:
        self.config = config
        self.git = GitManager(config.repos_dir)
        self.store = store if store is not None else VectorStore(config)
        self.providers = (
            providers if providers is not None else EmbeddingProviders(config)
        )
        self.states = (
            states
            if states is not None
            else StateStore(
                config.sqlite_index_path,
                config.state_dir / "repositories.json",
            )
        )
        self.pipeline = IndexPipeline(
            config, self.git, self.store, self.providers, self.states
        )
        self.retrieval = RetrievalService(
            config, self.git, self.store, self.providers, self.states
        )
        # Collections whose embedding model failed to load (degraded
        # startup; see ensure_collections / selfcheck).
        self._collection_errors: dict[CollectionName, str] = {}
        self._closed = False
        # Repository names with a sync (initial, periodic, polled, or
        # manual) currently in flight — see indexing_note().
        self._syncing: set[str] = set()

    # -- collections ---------------------------------------------------------

    def ensure_collections(self) -> None:
        """Create the vector-store tables (loads the embedding models).

        A model that fails to load (e.g. not present in the offline
        model cache) does not abort startup: the collection is left
        uncreated and the error recorded — the startup self-check
        reports it as degraded, and embedding search/indexing of that
        collection fails with a clear error until the model is
        provisioned (lexical search is unaffected).
        """
        dims: dict[CollectionName, int] = {}
        for collection in (
            CollectionName.HDL,
            CollectionName.DOCS,
            CollectionName.CODE,
        ):
            try:
                dims[collection] = self.providers.dimension(collection)
            except Exception as exc:
                logger.error(
                    "embedding model for the %s collection unavailable: %s; "
                    "the collection is degraded until the model is provisioned",
                    collection.value,
                    exc,
                )
                self._collection_errors[collection] = str(exc)
                dims[collection] = 0
        self.store.ensure_collections(
            hdl_dim=dims[CollectionName.HDL],
            docs_dim=dims[CollectionName.DOCS],
            code_dim=dims[CollectionName.CODE],
        )

    def collection_error(self, collection: CollectionName) -> str | None:
        """Why the collection's embedding model is unavailable, or None
        when it loaded."""
        return self._collection_errors.get(collection)

    def selfcheck(self) -> SelfCheckResult:
        """Run the startup self-check (after collections + migration)."""
        return run_self_check(self)

    def migrate_index(self) -> bool:
        """Migrate the index to the current schema layout (v1 -> v2).

        The SQLite store layout carries the schema version; a legacy
        store has the legacy ``vhdl`` collection dropped. Repository
        state now lives in the same database (created current at
        construction time); a legacy ``state/repositories.json``
        document is imported on first start. After any migration every
        repository's indexed commit is forgotten, so the next sync
        rebuilds the index deterministically from git (no manual data
        migration). Safe to call on every start: a current deployment
        is left untouched. Returns True when a migration ran.
        """
        db_migrated = self.store.migrate()
        state_migrated = self.states.migrate()
        if not db_migrated and not state_migrated:
            return False
        # A v1 -> v2 layout change invalidates the indexed commits.
        self.states.reset_all_indexed()
        logger.info(
            "index migrated to schema v%d (store: %s, state: %s); "
            "repositories reindex deterministically on the next sync",
            INDEX_SCHEMA_VERSION,
            db_migrated,
            state_migrated,
        )
        return True

    # -- sync -----------------------------------------------------------------

    async def sync_all(
        self, repositories: list[str] | None = None
    ) -> list[dict[str, str]]:
        """Synchronously update the selected repositories (default: all).

        Errors are contained per repository and reported, never raised.
        """
        wanted = set(repositories) if repositories is not None else None
        if wanted is not None:
            known = {cfg.name for cfg in self.config.repositories}
            if self.config.coding_standards is not None:
                known.add(CODING_STANDARDS_REPO)
            unknown = wanted - known
            if unknown:
                raise RetrievalError(
                    f"unknown repository: {', '.join(sorted(unknown))}"
                )
        reports: list[dict[str, str]] = []
        for cfg in self.config.repositories:
            if wanted is not None and cfg.name not in wanted:
                continue
            try:
                await self._tracked_sync(cfg)
                reports.append(
                    {
                        "repository": cfg.name,
                        "status": "ok",
                        "commit": self.states.get(cfg.name).indexed_commit or "",
                    }
                )
            except Exception as exc:
                logger.exception("%s: sync failed: %s", cfg.name, exc)
                reports.append(
                    {
                        "repository": cfg.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if self.config.coding_standards is not None and (
            wanted is None or CODING_STANDARDS_REPO in wanted
        ):
            report = sync_coding_standards(
                self.config, self.providers, self.store, self.states
            )
            if report is not None:
                reports.append(report)
        return reports

    async def reindex(self, repository: str) -> dict[str, str]:
        """Full reindex of one repository (error contained)."""
        if repository == CODING_STANDARDS_REPO:
            if self.config.coding_standards is None:
                raise RetrievalError("no coding_standards file is configured")
            report = sync_coding_standards(
                self.config, self.providers, self.store, self.states
            )
            if report is None:  # only when unconfigured; guarded above
                raise RetrievalError("no coding_standards file is configured")
            return report
        cfg = self._config_or_error(repository)
        try:
            await self._tracked_reindex(cfg)
        except Exception as exc:
            logger.exception("%s: reindex failed: %s", cfg.name, exc)
            return {"repository": cfg.name, "status": "error", "error": str(exc)}
        return {
            "repository": cfg.name,
            "status": "ok",
            "commit": self.states.get(cfg.name).indexed_commit or "",
        }

    async def _tracked_sync(self, cfg: RepositoryConfig) -> None:
        """Run ``pipeline.sync_repository`` while marking ``cfg.name`` as
        currently syncing (see ``indexing_note``)."""
        self._syncing.add(cfg.name)
        try:
            await self.pipeline.sync_repository(cfg)
        finally:
            self._syncing.discard(cfg.name)

    async def _tracked_reindex(self, cfg: RepositoryConfig) -> None:
        """Run ``pipeline.reindex_repository`` while marking ``cfg.name`` as
        currently syncing (see ``indexing_note``)."""
        self._syncing.add(cfg.name)
        try:
            await self.pipeline.reindex_repository(cfg)
        finally:
            self._syncing.discard(cfg.name)

    def indexing_note(self, repository: str | None = None) -> str | None:
        """A short heads-up for search tool output: whether ``repository``
        (or, if unset, any configured repository) is being (re)synced right
        now, or has never completed an initial sync at all. Lets the agent
        tell a thin/empty result set caused by in-progress (or stalled)
        indexing apart from a genuine no-match, instead of concluding
        nothing exists."""
        if repository is not None:
            names = [repository]
        else:
            names = [cfg.name for cfg in self.config.repositories]
            if self.config.coding_standards is not None:
                names.append(CODING_STANDARDS_REPO)
        syncing = sorted({name for name in names if name in self._syncing})
        never_synced = sorted(
            {
                name
                for name in names
                if name not in self._syncing
                and self.states.get(name).indexed_commit is None
            }
        )
        if not syncing and not never_synced:
            return None
        pending = set(syncing) | set(never_synced)
        auto = [
            cfg
            for cfg in self.config.repositories
            if cfg.auto_indexed and cfg.name in pending
        ]
        parts: list[str] = []
        if auto:
            derived = "; ".join(f"{cfg.name!r} ({cfg.path})" for cfg in auto)
            parts.append(
                "zero-config: no [[repositories]] are configured, so the "
                f"server auto-indexed its current directory as {derived}"
            )
        if syncing:
            parts.append(f"currently syncing: {', '.join(syncing)}")
        if never_synced:
            parts.append(
                f"not yet indexed: {', '.join(never_synced)} (initial sync "
                "still pending, or a prior sync failed — check "
                "repository_status)"
            )
        return (
            "Note: " + "; ".join(parts) + ". Results may be thin or "
            "incomplete; try again shortly."
        )

    def drop_unconfigured_repositories(self) -> list[str]:
        """Drop index chunks and state for repos removed from the config
        (the coding-standards pseudo-repository counts as configured when
        the ``coding_standards`` option is set)."""
        configured = {cfg.name for cfg in self.config.repositories}
        if self.config.coding_standards is not None:
            configured.add(CODING_STANDARDS_REPO)
        dropped = [
            state.name for state in self.states.all() if state.name not in configured
        ]
        for name in dropped:
            self.pipeline.delete_repository(name)
        return dropped

    async def periodic_sync(self) -> None:
        """Sync all repositories every ``config.sync_interval`` seconds."""
        interval = float(self.config.sync_interval)
        while True:
            await asyncio.sleep(interval)
            logger.info("periodic sync started (%.0fs interval)", interval)
            await self.sync_all()

    def _has_local_repos(self) -> bool:
        """True when at least one local working repository is configured."""
        return any(cfg.is_local for cfg in self.config.repositories)

    async def local_poll(self) -> None:
        """Fast change poller for local working repositories.

        Every ``local_sync_interval`` seconds, compute a cheap, read-only
        fingerprint (HEAD + porcelain status) for each local repository
        and run that repository's sync when the fingerprint differs from
        the one persisted at the last successful sync. This makes local
        work (commits, tracked edits, untracked file add/remove) show up
        in the index within about one interval instead of waiting for
        ``sync_interval``. Remote repositories are untouched (they still
        sync on ``sync_interval``). A sync failure is contained to the
        repository and never stops the poller.
        """
        interval = float(self.config.local_sync_interval)
        if interval <= 0 or not self._has_local_repos():
            return
        in_flight: set[str] = set()
        while True:
            await asyncio.sleep(interval)
            for cfg in self.config.repositories:
                if not cfg.is_local or cfg.name in in_flight:
                    continue
                try:
                    fingerprint = await self.git.local_fingerprint(cfg)
                except Exception as exc:  # contained per repository
                    logger.debug(
                        "%s: local poll could not fingerprint: %s",
                        cfg.name,
                        exc,
                    )
                    continue
                if fingerprint == self.states.get(cfg.name).local_fingerprint:
                    continue
                in_flight.add(cfg.name)
                logger.info("%s: local change detected (poll); syncing", cfg.name)
                task = asyncio.create_task(self._tracked_sync(cfg))
                # add_done_callback passes only the task; bind the per-repo
                # arguments with a partial so the loop variable ``cfg`` is
                # captured by value, not by reference.
                task.add_done_callback(
                    functools.partial(
                        _local_poll_done, name=cfg.name, in_flight=in_flight
                    )
                )

    # -- helpers -----------------------------------------------------------------

    def _config_or_error(self, name: str) -> RepositoryConfig:
        try:
            return self.config.repository(name)
        except ConfigError as exc:
            raise RetrievalError(str(exc)) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.states.close()
        self.store.close()


# -- MCP tools -----------------------------------------------------------------


def _render(
    results: list[SearchResult],
    empty: str,
    note: str | None = None,
    limit: int | None = None,
) -> str:
    body = "\n".join(result.render() for result in results) if results else empty
    if limit is not None and len(results) >= limit:
        body += (
            "\nNote: results may be truncated at the limit; increase "
            "`limit` or refine the query to see more."
        )
    return f"{note}\n\n{body}" if note else body


def _render_report(reports: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for report in reports:
        if report["status"] in ("ok", "up-to-date"):
            commit = report.get("commit", "")
            unchanged = " (unchanged)" if report["status"] == "up-to-date" else ""
            lines.append(
                f"- {report['repository']}: ok{unchanged}"
                + (f" (commit {commit[:12]})" if commit else "")
            )
        else:
            lines.append(
                f"- {report['repository']}: Error: {report.get('error', 'unknown')}"
            )
    return "\n".join(lines)


def _handle_errors[T: Callable[..., Awaitable[str]]](func: T) -> T:
    """Translate RetrievalError into a tool-level error message."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return await func(*args, **kwargs)
        except RetrievalError as exc:
            return f"Error: {exc}"

    return cast(T, wrapper)


def create_mcp(app: VhdlRagApp) -> MCPServer:
    """Create the MCPServer instance with all tools bound to ``app``."""
    mcp = MCPServer(MCP_NAME, instructions=INSTRUCTIONS)
    retrieval = app.retrieval

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_hdl(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        language: str | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Search HDL source — VHDL, Verilog, and SystemVerilog share one
        index: entities/modules (design units), architectures, processes
        and always blocks, packages, functions, tasks. Semantic +
        exact-identifier hybrid search. `language` restricts results to
        one language ('vhdl' | 'verilog' | 'systemverilog'); omit it to
        search all HDL. `symbols` restricts to chunks referencing the
        given identifiers (e.g. ["FIFO_DEPTH"]) — cross-referencing
        works across HDL languages. `repository` restricts to one
        repository name. `mode` selects the search strategy: 'hybrid'
        (default; semantic + full-text), 'semantic' (embedding
        similarity only), or 'lexical' (full-text match only)."""
        return _render(
            retrieval.search(
                CollectionName.HDL,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                language,
                mode=mode,
            ),
            "No HDL results. Try a broader query or a different language, "
            "or check repository_status.",
            note=app.indexing_note(repository),
            limit=limit,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_vhdl(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Back-compat alias for search_hdl(language="vhdl"). Prefer
        search_hdl directly — it covers VHDL plus Verilog/SystemVerilog and
        takes the same `query`/`limit`/`repository`/`symbols`/`mode`
        parameters (see its docstring for full parameter docs); this form
        is kept only for backward compatibility and has no advantage over
        it."""
        return _render(
            retrieval.search(
                CollectionName.HDL,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                "vhdl",
                mode=mode,
            ),
            "No VHDL results. Try a broader query, or check repository_status.",
            note=app.indexing_note(repository),
            limit=limit,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_docs(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Search VHDL-related documentation: coding standards, design
        guides, conventions (one result per section). `symbols` matches
        identifiers referenced in the section's code snippets. `mode`
        selects the search strategy: 'hybrid' (default; semantic +
        full-text), 'semantic' (embedding similarity only), or 'lexical'
        (full-text match only)."""
        return _render(
            retrieval.search(
                CollectionName.DOCS,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No documentation results. Try a broader query, or check "
            "repository_status.",
            note=app.indexing_note(repository),
            limit=limit,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_code(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Search general source code (C/C++, Python, ...): one result per
        function/class. `symbols` matches identifiers referenced in the
        unit (cross-reference to VHDL signal/port names, etc.). `mode`
        selects the search strategy: 'hybrid' (default; semantic +
        full-text), 'semantic' (embedding similarity only), or 'lexical'
        (full-text match only)."""
        return _render(
            retrieval.search(
                CollectionName.CODE,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No code results. Try a broader query, or check repository_status.",
            note=app.indexing_note(repository),
            limit=limit,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_knowledge(
        query: str,
        limit: int = KNOWLEDGE_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Search ALL domains (VHDL, documentation, code) at once, fused
        with RRF so the domains interleave fairly. Use when the question
        may span domains (e.g. a design requirement in the docs
        implemented in VHDL and tested in C). `mode` selects the search
        strategy: 'hybrid' (default; semantic + full-text), 'semantic'
        (embedding similarity only), or 'lexical' (full-text match
        only)."""
        return _render(
            retrieval.search_knowledge(
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No results in any domain. Try a broader query, or check "
            "repository_status.",
            note=app.indexing_note(repository),
            limit=limit,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def get_source(
        repository: str,
        file: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read the exact current content of an indexed file (or a line
        range) from the synced repository, with commit attribution.
        `file` is the repository-relative path from any search result's
        source line."""
        return retrieval.get_source(repository, file, start_line, end_line)

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def repository_files(
        repository: str,
        pattern: str | None = None,
        limit: int = 200,
    ) -> str:
        """List the files known to the index for a repository — the
        candidate paths to pass to get_source (no guessing). `pattern` is
        a glob matched against the repository-relative path ('*' crosses
        '/'), e.g. 'modules/counter/*' or '*.vhd'. Results are capped at
        `limit`; a truncation note is appended when more exist."""
        files, truncated = retrieval.list_files(repository, pattern, limit)
        if not files:
            return (
                f"No indexed files in {repository!r} match "
                f"({pattern or 'any pattern'}). Check repository_status — "
                "the repository may not be synced yet."
            )
        out = "\n".join(files)
        if truncated:
            out += f"\n… (capped at {limit}; refine the pattern for more)"
        return out

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def repository_status() -> str:
        """Show every configured repository: ref, enabled domains, last
        indexed commit, chunk and file counts, last sync time, and any
        sync error. When a coding-standards file is configured it is
        shown as the 'coding-standards' pseudo-repository (its content
        hash in place of a commit). Also reports the HDL analyzers
        (vhdl_ls for VHDL, Veridian for Verilog/SystemVerilog):
        availability, version, and whether semantic (lsp) or fallback
        parsing is in effect."""
        lines: list[str] = []
        for status in retrieval.repository_status():
            domains = ", ".join(status.domains)
            commit = status.indexed_commit[:12] if status.indexed_commit else "never"
            if status.filesystem:
                # Filesystem repositories attribute chunks to the walk
                # fingerprint, not a commit.
                commit = f"{commit} (fingerprint)" if status.indexed_commit else "never"
            source = "filesystem" if status.filesystem else f"ref {status.ref}"
            synced = status.last_sync_at or "never"
            error = (
                f"\n  last error: {status.last_sync_error}"
                if status.last_sync_error is not None
                else ""
            )
            per_domain: list[str] = []
            for domain in status.domains:
                count = app.store.count_repository(status.name, CollectionName(domain))
                per_domain.append(f"{count} {domain}")
            total = app.store.count_repository(status.name)
            lines.append(
                f"- {status.name} ({source}, priority {status.priority}, "
                f"domains: {domains})\n"
                f"  indexed: {commit}, synced: {synced}\n"
                f"  chunks: {' + '.join(per_domain)} ({total} total), "
                f"files: {status.file_count}{error}"
            )
        standards_line: str | None = None
        if app.config.coding_standards is not None:
            state = app.states.get(CODING_STANDARDS_REPO)
            commit = state.indexed_commit[:12] if state.indexed_commit else "never"
            synced = state.last_sync_at.isoformat() if state.last_sync_at else "never"
            error = (
                f"\n  last error: {state.last_sync_error}"
                if state.last_sync_error is not None
                else ""
            )
            chunks = app.store.count_repository(CODING_STANDARDS_REPO)
            standards_line = (
                f"- {CODING_STANDARDS_REPO} (file {app.config.coding_standards}, "
                f"priority {app.config.coding_standards_priority})\n"
                f"  indexed: {commit}, synced: {synced}\n"
                f"  chunks: {chunks} docs{error}"
            )
        if not lines and standards_line is None:
            return "No repositories configured."
        if lines:
            lines.append("")
        if standards_line is not None:
            lines.append(standards_line)
            lines.append("")
        lines.append("HDL analyzers:")
        for analyzer in build_analyzer_statuses(
            app.config.vhdl_ls_path, app.config.veridian_path
        ).values():
            if analyzer.available:
                line = f"- {analyzer.name}: {analyzer.mode}, {analyzer.version}"
                if analyzer.path:
                    line += f" ({analyzer.path})"
            else:
                line = f"- {analyzer.name}: {analyzer.mode} — {analyzer.error}"
            lines.append(line)
        lines.append("")
        lines.append("Embedding models:")
        for collection in ALL_COLLECTIONS:
            model_error = app.collection_error(collection)
            if model_error is None:
                lines.append(
                    f"- {collection.value}: {app.providers.model_name(collection)}"
                )
            else:
                lines.append(f"- {collection.value}: unavailable — {model_error}")
        return "\n".join(lines)

    @mcp.tool(annotations=_READ_WRITE)
    @_handle_errors
    async def sync_repositories(repositories: list[str] | None = None) -> str:
        """Incrementally sync repositories (default: all): fetch the ref,
        chunk changed files, update the index. Safe to call any time;
        failures are contained per repository and reported."""
        reports = await app.sync_all(repositories)
        return _render_report(reports)

    @mcp.tool(annotations=_READ_WRITE)
    @_handle_errors
    async def reindex_repository(repository: str) -> str:
        """Fully reindex one repository (drops and rebuilds all of its
        chunks). Use after config changes or to repair a drifted index."""
        report = await app.reindex(repository)
        return _render_report([report])

    return mcp


def _acquire_lock(config: AppConfig) -> Path:
    """Take an exclusive single-instance lock; exit on failure.

    POSIX uses ``flock(2)`` (per open file description); Windows uses an
    advisory byte-range lock via ``msvcrt`` (per handle). The lock file
    records the owning PID for diagnosis.
    """
    lock_path = config.resolved_data_dir / "server.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # Windows has no flock(2); use an advisory byte-range lock on the
        # first byte (held per handle, released on close/exit).
        handle = open(lock_path, "a+b")  # noqa: SIM115
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logger.error(
                "another corvidex-mcp instance holds the lock (%s); exiting",
                lock_path,
            )
            raise SystemExit(1) from None
        handle.write(f"{os.getpid()}\n".encode())
    else:
        handle = open(lock_path, "a+")  # noqa: SIM115
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.error(
                "another corvidex-mcp instance holds the lock (%s); exiting",
                lock_path,
            )
            raise SystemExit(1) from None
        handle.write(f"{os.getpid()}\n")
    handle.flush()
    # Keep the handle open for the process lifetime.
    global _LOCK_HANDLE
    _LOCK_HANDLE = handle
    return lock_path


# -- startup -----------------------------------------------------------------


async def _serve(
    app: VhdlRagApp, mcp: MCPServer, initial_sync: asyncio.Task[list[dict[str, str]]]
) -> None:
    """Serve stdio with background sync tasks: the initial sync (started
    before this is called, so it never delays the MCP handshake — tools
    called while it is still running see an indexing_note() heads-up),
    the periodic sync (all repositories), and the fast change poller
    (local working repos)."""
    sync_task = asyncio.create_task(app.periodic_sync())
    poll_task = (
        asyncio.create_task(app.local_poll()) if app._has_local_repos() else None
    )
    try:
        await mcp.run_stdio_async()
    finally:
        initial_sync.cancel()
        sync_task.cancel()
        if poll_task is not None:
            poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await initial_sync
        with contextlib.suppress(asyncio.CancelledError):
            await sync_task
        if poll_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
        app.close()


async def _main_async(app: VhdlRagApp, mcp: MCPServer) -> None:
    logger.info("ensuring collections (embedding models load from the local cache)")
    app.ensure_collections()
    app.migrate_index()
    check = app.selfcheck()
    if not check.required_ok:
        for component in check.components:
            if not component.ok and not component.optional:
                logger.error(
                    "startup self-check %s: %s", component.name, component.detail
                )
        logger.error(
            "startup self-check failed (%s); not serving. Fix the missing "
            "component and restart.",
            ", ".join(check.degraded),
        )
        raise SystemExit(1)
    logger.info("startup self-check: %s", check.summary())
    dropped = app.drop_unconfigured_repositories()
    if dropped:
        logger.info(
            "dropped chunks of unconfigured repositories: %s", ", ".join(dropped)
        )
    if not app.config.repositories and app.config.coding_standards is None:
        logger.warning(
            "no repositories and no coding_standards file configured: the "
            "index stays empty. Add [[repositories]] entries (or a "
            "coding_standards file) to the config file "
            "($CORVIDEX_MCP_CONFIG or ~/.config/corvidex/config.toml) and "
            "restart, or call sync_repositories after updating it."
        )
    logger.info(
        "starting initial sync of %d repositories in the background "
        "(the server starts serving immediately; tools report an "
        "indexing note for repositories not yet synced)",
        len(app.config.repositories),
    )
    initial_sync = asyncio.create_task(app.sync_all())
    await _serve(app, mcp, initial_sync)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="corvidex-mcp",
        description="MCP server: semantic search over VHDL repositories.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "config file (default: $CORVIDEX_MCP_CONFIG or "
            "~/.config/corvidex/config.toml)"
        ),
    )
    parser.add_argument(
        "--data-dir", default=None, metavar="PATH", help="override data_dir"
    )
    parser.add_argument(
        "--sync-interval",
        default=None,
        type=int,
        metavar="SECONDS",
        help="override sync_interval",
    )
    parser.add_argument(
        "--local-sync-interval",
        default=None,
        type=int,
        metavar="SECONDS",
        help="override local_sync_interval (fast poller for local repos)",
    )
    parser.add_argument(
        "--vhdl-ls-path",
        default=None,
        metavar="PATH",
        help="override vhdl_ls_path",
    )
    parser.add_argument(
        "--veridian-path",
        default=None,
        metavar="PATH",
        help="override veridian_path",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="override log_level",
    )
    parser.add_argument(
        "--no-index-cwd",
        action="store_true",
        help=(
            "disable index_cwd: do not automatically index the directory "
            "the server is started in when no [[repositories]] are "
            "configured (run with an empty index instead)"
        ),
    )
    parser.add_argument(
        "--num-threads",
        default=None,
        type=int,
        metavar="N",
        help=("override embeddings.dense_threads (default: half the host's CPU count)"),
    )
    return parser.parse_args(argv)


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    """Parse CLI arguments and load the config with overrides applied.

    The config file is selected by ``--config`` or
    ``CORVIDEX_MCP_CONFIG``; ``--data-dir``/``--sync-interval``/
    ``--vhdl-ls-path``/``--veridian-path``/``--log-level``/
    ``--num-threads`` override the file's values.
    """
    args = _parse_args(argv)
    config = load_config(
        Path(args.config) if args.config else None,
        inject_default_repository=False,
    )
    overrides: dict[str, Any] = {
        field: getattr(args, field)
        for field in _CLI_SCALAR_OVERRIDES
        if getattr(args, field) is not None
    }
    if args.no_index_cwd:
        overrides["index_cwd"] = False
    if overrides or args.num_threads is not None:
        # Re-validate from the *explicitly set* fields only: a full
        # model_dump() would mark every default as user-set, which
        # apply_default_repository() relies on (model_fields_set) to
        # decide whether data_dir may be redirected per project.
        raw = config.model_dump(exclude_unset=True)
        raw.update(overrides)
        if args.num_threads is not None:
            raw.setdefault("embeddings", {})["dense_threads"] = args.num_threads
        config = AppConfig.model_validate(raw)
    return apply_default_repository(config)


#: Top-level scalar CLI flags whose argparse dest equals the AppConfig field.
_CLI_SCALAR_OVERRIDES = (
    "data_dir",
    "sync_interval",
    "local_sync_interval",
    "vhdl_ls_path",
    "veridian_path",
    "log_level",
)


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server over stdio (uvx entry point)."""
    try:
        config = config_from_args(argv)
    except (ConfigError, ValidationError) as exc:
        print(f"corvidex-mcp: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    setup_logging(config.log_level, config.log_file)
    logger.info(
        "corvidex-mcp starting (data_dir=%s, %d repositories)",
        config.resolved_data_dir,
        len(config.repositories),
    )
    _acquire_lock(config)
    app = VhdlRagApp(config)
    mcp = create_mcp(app)
    try:
        asyncio.run(_main_async(app, mcp))
    finally:
        app.close()


if __name__ == "__main__":
    main()
