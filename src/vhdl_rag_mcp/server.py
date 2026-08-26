"""FastMCP server: VHDL RAG search over configured Git repositories.

Exposes eight tools to coding agents:

- ``search_vhdl`` / ``search_docs`` / ``search_code`` — hybrid
  (dense + sparse) semantic search in one domain, with optional
  repository/category filters and identifier cross-references;
- ``search_knowledge`` — the same search fused across all three
  domains (RRF over the per-domain rank lists);
- ``get_source`` — exact file content (or a line range) from the
  synced working tree, with repository/commit attribution;
- ``repository_status`` — what is indexed and any sync errors;
- ``sync_repositories`` / ``reindex_repository`` — maintenance
  (incremental sync of selected repos, full reindex of one).

Lifecycle: load config, log to stderr/file (stdout is reserved for
the MCP protocol), take a single-instance lock, create the Qdrant
collections (downloading the embedding models on first run), run an
initial sync, then serve stdio while a background task syncs every
``sync_interval`` seconds. All failures are contained per repository:
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

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from .config import AppConfig, ConfigError, RepositoryConfig, load_config
from .embeddings.providers import EmbeddingProviders
from .git_manager import GitManager
from .indexing import IndexPipeline
from .logging_setup import setup_logging
from .lsp import build_analyzer_statuses
from .models import INDEX_SCHEMA_VERSION, CollectionName, SearchResult
from .retrieval import RetrievalError, RetrievalService
from .state import StateStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Held open for the process lifetime while the lock is taken.
_LOCK_HANDLE: object | None = None

MCP_NAME = "vhdl_rag_mcp"

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
    "languages). Use get_source for the full text of a known file (exact "
    "lines, exact commit). Every result carries source attribution "
    "(repository, file, line range, commit, language). Use "
    "repository_status to see what is indexed and whether a sync failed; "
    "it also reports the HDL analyzer status (vhdl_ls / Veridian). "
    "sync_repositories to force an update, reindex_repository to rebuild "
    "one repository's index."
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True)
_READ_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False)

DEFAULT_LIMIT = 8
KNOWLEDGE_LIMIT = 10


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
            else StateStore(config.state_dir / "repositories.json")
        )
        self.pipeline = IndexPipeline(
            config, self.git, self.store, self.providers, self.states
        )
        self.retrieval = RetrievalService(
            config, self.git, self.store, self.providers, self.states
        )
        self._closed = False

    # -- collections ---------------------------------------------------------

    def ensure_collections(self) -> None:
        """Create the Qdrant collections (loads the embedding models)."""
        self.store.ensure_collections(
            hdl_dim=self.providers.dimension(CollectionName.HDL),
            docs_dim=self.providers.dimension(CollectionName.DOCS),
            code_dim=self.providers.dimension(CollectionName.CODE),
        )

    def migrate_index(self) -> bool:
        """Migrate the index to the current schema layout (v1 -> v2).

        The legacy ``vhdl`` collection is dropped and every repository's
        indexed commit is forgotten, so the next sync rebuilds the index
        deterministically from git (no manual data migration). Safe to
        call on every start: a current document is left untouched.
        Returns True when a migration ran.
        """
        if not self.states.needs_migration:
            return False
        dropped = self.store.delete_legacy_vhdl()
        self.states.migrate()
        logger.info(
            "index migrated to schema v%d (legacy vhdl collection dropped: %s); "
            "repositories reindex deterministically on the next sync",
            INDEX_SCHEMA_VERSION,
            dropped,
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
            unknown = wanted - {cfg.name for cfg in self.config.repositories}
            if unknown:
                raise RetrievalError(
                    f"unknown repository: {', '.join(sorted(unknown))}"
                )
        reports: list[dict[str, str]] = []
        for cfg in self.config.repositories:
            if wanted is not None and cfg.name not in wanted:
                continue
            try:
                await self.pipeline.sync_repository(cfg)
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
        return reports

    async def reindex(self, repository: str) -> dict[str, str]:
        """Full reindex of one repository (error contained)."""
        cfg = self._config_or_error(repository)
        try:
            await self.pipeline.reindex_repository(cfg)
        except Exception as exc:
            logger.exception("%s: reindex failed: %s", cfg.name, exc)
            return {"repository": cfg.name, "status": "error", "error": str(exc)}
        return {
            "repository": cfg.name,
            "status": "ok",
            "commit": self.states.get(cfg.name).indexed_commit or "",
        }

    def drop_unconfigured_repositories(self) -> list[str]:
        """Drop index chunks and state for repos removed from the config."""
        configured = {cfg.name for cfg in self.config.repositories}
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
        self.store.close()


# -- MCP tools -----------------------------------------------------------------


def _render(results: list[SearchResult], empty: str) -> str:
    if not results:
        return empty
    return "\n".join(result.render() for result in results)


def _render_report(reports: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for report in reports:
        if report["status"] == "ok":
            commit = report.get("commit", "")
            lines.append(
                f"- {report['repository']}: ok"
                + (f" (commit {commit[:12]})" if commit else "")
            )
        else:
            lines.append(
                f"- {report['repository']}: ERROR: {report.get('error', 'unknown')}"
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


def create_mcp(app: VhdlRagApp) -> FastMCP:
    """Create the FastMCP instance with all tools bound to ``app``."""
    mcp = FastMCP(MCP_NAME, instructions=INSTRUCTIONS)
    retrieval = app.retrieval

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_hdl(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
        language: str | None = None,
    ) -> str:
        """Search HDL source — VHDL, Verilog, and SystemVerilog share one
        index: entities/modules (design units), architectures, processes
        and always blocks, packages, functions, tasks. Semantic +
        exact-identifier hybrid search. `language` restricts results to
        one language ('vhdl' | 'verilog' | 'systemverilog'); omit it to
        search all HDL. `symbols` restricts to chunks referencing the
        given identifiers (e.g. ["FIFO_DEPTH"]) — cross-referencing
        works across HDL languages. `repository` restricts to one
        repository name."""
        return _render(
            retrieval.search(
                CollectionName.HDL,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                language,
            ),
            "No HDL results. Try a broader query or a different language, "
            "or check repository_status.",
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_vhdl(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
    ) -> str:
        """Search VHDL source only (the language-restricted form of
        search_hdl; use search_hdl for Verilog/SystemVerilog or a mix):
        entities, architectures, processes, packages, functions —
        semantic + exact-identifier hybrid search. `symbols` restricts
        to chunks referencing the given identifiers (e.g.
        ["fifo_write", "rst_n"]). `repository` restricts to one
        repository name."""
        return _render(
            retrieval.search(
                CollectionName.HDL,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                "vhdl",
            ),
            "No VHDL results. Try a broader query, or check repository_status.",
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_docs(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
    ) -> str:
        """Search VHDL-related documentation: coding standards, design
        guides, conventions (one result per section). `symbols` matches
        identifiers referenced in the section's code snippets."""
        return _render(
            retrieval.search(
                CollectionName.DOCS,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
            ),
            "No documentation results. Try a broader query, or check "
            "repository_status.",
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_code(
        query: str,
        limit: int = DEFAULT_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
    ) -> str:
        """Search general source code (C/C++, Python, ...): one result per
        function/class. `symbols` matches identifiers referenced in the
        unit (cross-reference to VHDL signal/port names, etc.)."""
        return _render(
            retrieval.search(
                CollectionName.CODE,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
            ),
            "No code results. Try a broader query, or check repository_status.",
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_knowledge(
        query: str,
        limit: int = KNOWLEDGE_LIMIT,
        repository: str | None = None,
        symbols: list[str] | None = None,
    ) -> str:
        """Search ALL domains (VHDL, documentation, code) at once, fused
        with RRF so the domains interleave fairly. Use when the question
        may span domains (e.g. a design requirement in the docs
        implemented in VHDL and tested in C)."""
        return _render(
            retrieval.search_knowledge(
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
            ),
            "No results in any domain. Try a broader query, or check "
            "repository_status.",
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
    async def repository_status() -> str:
        """Show every configured repository: ref, enabled domains, last
        indexed commit, chunk and file counts, last sync time, and any
        sync error. Also reports the HDL analyzers (vhdl_ls for VHDL,
        Veridian for Verilog/SystemVerilog): availability, version, and
        whether semantic (lsp) or fallback parsing is in effect."""
        lines: list[str] = []
        for status in retrieval.repository_status():
            domains = ", ".join(status.domains)
            commit = status.indexed_commit[:12] if status.indexed_commit else "never"
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
                f"- {status.name} (ref {status.ref}, domains: {domains})\n"
                f"  indexed: {commit}, synced: {synced}\n"
                f"  chunks: {' + '.join(per_domain)} ({total} total), "
                f"files: {status.file_count}{error}"
            )
        if not lines:
            return "No repositories configured."
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
                "another vhdl-rag-mcp instance holds the lock (%s); exiting",
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
                "another vhdl-rag-mcp instance holds the lock (%s); exiting",
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


async def _serve(app: VhdlRagApp, mcp: FastMCP) -> None:
    """Serve stdio with a background periodic-sync task."""
    sync_task = asyncio.create_task(app.periodic_sync())
    try:
        await mcp.run_stdio_async()
    finally:
        sync_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sync_task
        app.close()


async def _main_async(app: VhdlRagApp, mcp: FastMCP) -> None:
    logger.info("ensuring collections (embedding models download on first run)")
    app.ensure_collections()
    app.migrate_index()
    dropped = app.drop_unconfigured_repositories()
    if dropped:
        logger.info(
            "dropped chunks of unconfigured repositories: %s", ", ".join(dropped)
        )
    logger.info("initial sync of %d repositories", len(app.config.repositories))
    await app.sync_all()
    await _serve(app, mcp)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vhdl-rag-mcp",
        description="MCP server: semantic search over VHDL repositories.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "config file (default: $VHDL_RAG_MCP_CONFIG or "
            "~/.config/vhdl-rag/config.toml)"
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
    return parser.parse_args(argv)


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    """Parse CLI arguments and load the config with overrides applied.

    The config file is selected by ``--config`` or
    ``VHDL_RAG_MCP_CONFIG``; ``--data-dir``/``--sync-interval``/
    ``--vhdl-ls-path``/``--veridian-path``/``--log-level`` override the
    file's values.
    """
    args = _parse_args(argv)
    config = load_config(Path(args.config) if args.config else None)
    overrides: dict[str, Any] = {}
    if args.data_dir is not None:
        overrides["data_dir"] = Path(args.data_dir)
    if args.sync_interval is not None:
        overrides["sync_interval"] = args.sync_interval
    if args.vhdl_ls_path is not None:
        overrides["vhdl_ls_path"] = args.vhdl_ls_path
    if args.veridian_path is not None:
        overrides["veridian_path"] = args.veridian_path
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if overrides:
        raw = config.model_dump()
        raw.update(overrides)
        config = AppConfig.model_validate(raw)
    return config


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server over stdio (uvx entry point)."""
    try:
        config = config_from_args(argv)
    except (ConfigError, ValidationError) as exc:
        print(f"vhdl-rag-mcp: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    setup_logging(config.log_level, config.log_file)
    logger.info(
        "vhdl-rag-mcp starting (data_dir=%s, %d repositories)",
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
