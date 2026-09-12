"""MCP server: RTL-centric RAG and indexing over configured Git repositories.

Exposes thirteen tools to coding agents, in two families.

Fuzzy retrieval (expensive, answers "I don't know the name"):

- ``search_hdl`` / ``search_docs`` / ``search_code`` — hybrid
  (dense + full-text) semantic search in one domain, with optional
  repository/language filters and identifier cross-references;
- ``search_knowledge`` — the same search fused across all three
  domains (RRF over the per-domain rank lists).

Exact lookup (cheap, answers "I know the name or the position" — see
:mod:`corvidex_mcp.navigation`):

- ``find_symbol`` — the language server's own workspace symbol lookup
  by name (the cheapest way to locate a known identifier);
- ``find_definition`` / ``find_references`` / ``hover_info`` — LSP
  go-to-definition / find-references / hover at a known position.

Plus exact reads and maintenance:

- ``get_source`` — exact file content (or a line range) from the
  synced working tree, with repository/commit attribution;
- ``repository_files`` — list the indexed files of a repository
  (glob-filterable), i.e. the candidate paths for ``get_source``;
- ``repository_status`` — what is indexed, what is syncing, and any
  sync errors;
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
from typing import Annotated, Any, cast

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
from pydantic import Field, ValidationError

from .config import (
    CODING_STANDARDS_REPO,
    DEFAULT_TEMPLATE,
    PROJECT_DIR_ENV,
    AppConfig,
    ConfigError,
    RepositoryConfig,
    apply_default_repository,
    load_config,
    project_config_path,
)
from .embeddings.providers import EmbeddingProviders
from .git_manager import GitManager
from .indexing import IndexPipeline
from .logging_setup import setup_logging
from .lsp import AnalyzerStatus, build_analyzer_statuses
from .models import INDEX_SCHEMA_VERSION, CollectionName, SearchResults
from .navigation import REFERENCE_LIMIT
from .navigation import find_definition as _find_definition
from .navigation import find_references as _find_references
from .navigation import find_symbol as _find_symbol
from .navigation import hover_info as _hover_info
from .retrieval import WEAK_MATCH_SCORE, RetrievalError, RetrievalService
from .selfcheck import SelfCheckResult, run_self_check
from .standards import sync_coding_standards
from .state import StateStore
from .vector_store import ALL_COLLECTIONS, VectorStore

logger = logging.getLogger(__name__)

#: Held open for the process lifetime while the lock is taken.
_LOCK_HANDLE: object | None = None

MCP_NAME = "corvidex_mcp"

INSTRUCTIONS = (
    "WHAT THIS IS. An index of the organization's HDL (VHDL, Verilog, "
    "SystemVerilog), design documentation and coding standards, and "
    "general source code (C/C++, Python, ...) — including repositories "
    "outside the working tree, every result attributed to a repository, "
    "file, line range and commit. Prefer it over reading/grepping the "
    "working tree when the answer may live in another repository, when "
    "the question is conceptual rather than a literal string, or when "
    "convention is the answer (a configured coding-standards file is "
    "indexed as the 'coding-standards' pseudo-repository at high "
    "priority). grep/Read stay better for a literal string in a file "
    "you already have open.\n"
    "ROUTING, cheapest and most exact first. (1) Exact identifier, want "
    "its declaration: find_symbol. (2) Known file:line:character, want "
    "the declaration, the use sites or the type: find_definition, "
    "find_references, hover_info (LSP-backed; positions 0-based, "
    "results rendered 1-based as path:line:col). Those four are exact "
    "and cost ~100-400 tokens. (3) A concept, a natural-language "
    "question, or no name at all: search_hdl / search_docs / "
    "search_code, or search_knowledge when the question spans docs + "
    "RTL + tests. Every search hit returns a whole indexed construct, "
    "so a search commonly costs one to two orders of magnitude more "
    "than a navigation call and answers less precisely — never search "
    "for an identifier you already know. (4) Known file, want its text: "
    "get_source, not a search. (5) Unknown path: repository_files — "
    "don't guess.\n"
    "RESULTS. A search hit quotes at most 40 lines of the matched "
    "chunk; when more were elided the last body line is the exact "
    "get_source call that returns them. Quoted lines carry a 1-based "
    "line-number gutter (the same numbering get_source uses), while "
    "find_definition/find_references/hover_info take 0-based lines — "
    "pass N-1 for a line displayed as N. `score` is the cross-encoder "
    "reranker's relevance in 0-1 when reranking is available "
    "(comparable across queries; below ~0.05 means no real match and "
    "the search says so), otherwise the store's rank-fused score, "
    "which is only comparable within one response.\n"
    "CAVEATS. `symbols` restricts a search to chunks referencing given "
    "identifiers: the key for tracing one name across docs, RTL and "
    "testbenches, and it works across HDL languages. `mode` is 'hybrid' "
    "(default), 'semantic', or 'lexical' (full-text only; the only mode "
    "needing no embedding model). repository_status is the source of "
    "truth for repository names, for whether a sync is in progress or "
    "has failed, and for analyzer (vhdl_ls / Veridian) status; a "
    "zero-config repository name carries a hash suffix (e.g. "
    "'vhdl-ai-test-582e8509'), so read it there rather than guessing. A "
    "search result may open with a 'Note:' line naming repositories "
    "'currently syncing' or 'not yet indexed': results stay thin until "
    "that finishes, so retry instead of concluding nothing exists. "
    "sync_repositories and reindex_repository exist for repair; the "
    "index syncs itself."
)

_READ_ONLY = ToolAnnotations(read_only_hint=True)
_READ_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)

DEFAULT_LIMIT = 8
KNOWLEDGE_LIMIT = 10

# -- parameter descriptions ----------------------------------------------------
#
# ``tools/list`` shows a parameter as a bare name and type unless the
# annotation carries a pydantic Field description, so everything an
# agent can learn about an argument without calling the tool has to live
# here (the docstring alone is prose it cannot map onto arguments).
# Shared across tools where the meaning is genuinely identical.

QueryArg = Annotated[
    str,
    Field(
        description=(
            "What to search for, in the words the code or docs would use "
            "('handshake on a full FIFO', 'asynchronous reset "
            "convention'). Natural language and identifier fragments both "
            "work. If you already know the exact identifier, call "
            "find_symbol instead — it is exact and far cheaper."
        )
    ),
]

LimitArg = Annotated[
    int,
    Field(
        description=(
            "Maximum number of results (default 8; search_knowledge "
            "defaults to 10). Each result is a whole construct or "
            "section, so raising this costs real tokens — raise it only "
            "after a truncation note says more matches exist."
        )
    ),
]

KnowledgeLimitArg = Annotated[
    int,
    Field(
        description=(
            "Maximum number of results (default 10 here, 8 for the "
            "single-domain search tools) shared across all three domains "
            "after the RRF fusion."
        )
    ),
]

RepositoryFilterArg = Annotated[
    str | None,
    Field(
        description=(
            "Restrict the search to one repository by its exact name "
            "(default: every configured repository). Names come from "
            "repository_status — in zero-config mode a name carries a "
            "hash suffix (e.g. 'vhdl-ai-test-582e8509'), so never guess "
            "one. 'coding-standards' restricts to the configured "
            "coding-standards file."
        )
    ),
]

SymbolsArg = Annotated[
    list[str] | None,
    Field(
        description=(
            "Restrict results to chunks that reference these exact "
            "identifiers, e.g. ['FIFO_DEPTH'] (default: no identifier "
            "restriction). This is the cross-domain tracing key: the same "
            "name matches in RTL, testbenches, docs and C code, and "
            "across HDL languages. Use it to follow one signal/generic/"
            "constant through the system; drop it for conceptual queries."
        )
    ),
]

ModeArg = Annotated[
    str,
    Field(
        description=(
            "Search strategy. 'hybrid' (default) fuses embedding "
            "similarity with full-text match and is right for almost "
            "everything. 'semantic' is embeddings only — for a paraphrase "
            "that shares no vocabulary with the code. 'lexical' is "
            "full-text only — exact words, spellings and error strings, "
            "and the only mode that needs no embedding model (so it still "
            "works when repository_status reports one unavailable)."
        )
    ),
]

LanguageArg = Annotated[
    str | None,
    Field(
        description=(
            "Restrict to one HDL language: 'vhdl', 'verilog' or "
            "'systemverilog' (default: all three, which share one index). "
            "Use it only when the other languages would be noise — a "
            "mixed-language design is usually best searched whole."
        )
    ),
]

RepositoryArg = Annotated[
    str,
    Field(
        description=(
            "Repository name, exactly as repository_status reports it "
            "(zero-config names carry a hash suffix, e.g. "
            "'vhdl-ai-test-582e8509')."
        )
    ),
]

FileArg = Annotated[
    str,
    Field(
        description=(
            "Repository-relative path, as printed on a search result's "
            "source line or by repository_files. Never an absolute path, "
            "and never guessed — call repository_files if unsure."
        )
    ),
]

LineArg = Annotated[
    int,
    Field(
        description=(
            "0-BASED line of the symbol (LSP convention): the first line "
            "of a file is 0, and a line displayed as N in a 1-based "
            "listing is passed here as N-1."
        )
    ),
]

CharacterArg = Annotated[
    int,
    Field(
        description=(
            "0-BASED column of the symbol on that line: point at the "
            "identifier itself, not at the start of the line."
        )
    ),
]


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
        # HDL analyzer probes (vhdl_ls, Veridian) — spawns a subprocess
        # per analyzer, so this is memoized on first access instead of
        # re-probed on every repository_status() call or self-check; see
        # analyzer_statuses().
        self._analyzer_statuses: dict[str, AnalyzerStatus] | None = None

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

    def analyzer_statuses(self) -> dict[str, AnalyzerStatus]:
        """HDL analyzer probes (vhdl_ls, Veridian), probed once per
        process and cached — reused by both the startup self-check and
        the repository_status tool."""
        if self._analyzer_statuses is None:
            self._analyzer_statuses = build_analyzer_statuses(
                self.config.vhdl_ls_path, self.config.veridian_path
            )
        return self._analyzer_statuses

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
            known = set(self.config.configured_repository_names())
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
            report = await self._tracked_standards_sync()
            if report is not None:
                reports.append(report)
        return reports

    async def reindex(self, repository: str) -> dict[str, str]:
        """Full reindex of one repository (error contained)."""
        if repository == CODING_STANDARDS_REPO:
            if self.config.coding_standards is None:
                raise RetrievalError("no coding_standards file is configured")
            report = await self._tracked_standards_sync()
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

    async def _tracked_standards_sync(self) -> dict[str, str] | None:
        """Run ``sync_coding_standards`` while marking the
        coding-standards pseudo-repository as currently syncing, so
        ``indexing_note``/``sync_state`` report it like any other
        repository (it fails permanently when the configured file is
        missing, which must not read as an in-progress index)."""
        self._syncing.add(CODING_STANDARDS_REPO)
        try:
            return await sync_coding_standards(
                self.config, self.providers, self.store, self.states
            )
        finally:
            self._syncing.discard(CODING_STANDARDS_REPO)

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
            names = list(self.config.configured_repository_names())
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

    def sync_state(
        self, repository: str, indexed_commit: str | None, last_sync_error: str | None
    ) -> str:
        """One line saying whether ``repository`` is syncing *right now*,
        has permanently failed, or is idle — for ``repository_status``.

        Without it an initial index (minutes, for a large repository)
        looks exactly like a broken one: "indexed: never, synced: never,
        files: 0", with chunk counts quietly growing between calls and
        nothing anywhere saying "in progress". And a repository whose
        sync fails every cycle for a permanent reason (a missing
        coding-standards file, a ref that does not resolve) looks the
        same again, because a retry is scheduled forever. The caller
        needs to tell "wait and retry" from "fix something".
        """
        if repository in self._syncing:
            if indexed_commit is None:
                return (
                    "IN PROGRESS — initial index running; the counts below "
                    "grow as it proceeds and searches stay thin until it "
                    "finishes. Retry in a few seconds."
                )
            return (
                "IN PROGRESS — resync running; the previously indexed "
                "commit below stays queryable meanwhile"
            )
        retry = f"retried every {self.config.sync_interval}s"
        if last_sync_error is not None:
            if indexed_commit is None:
                return (
                    f"FAILED — nothing has ever been indexed, and the {retry} "
                    "retry will keep failing until the cause under 'last "
                    "error' is fixed (this is not an in-progress sync)"
                )
            return (
                f"FAILED — serving the last good index below; {retry} (see "
                "'last error')"
            )
        if indexed_commit is None:
            return "pending — queued for the next sync cycle, nothing indexed yet"
        return "idle — up to date"

    def drop_unconfigured_repositories(self) -> list[str]:
        """Drop index chunks and state for repos removed from the config
        (the coding-standards pseudo-repository counts as configured when
        the ``coding_standards`` option is set)."""
        configured = set(self.config.configured_repository_names())
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


def _weak_match_hint(results: SearchResults) -> str | None:
    """One line warning that the best hit is a poor match, or None.

    Only emitted when the scores are cross-encoder relevance in
    ``(0, 1)`` (see :attr:`SearchResults.calibrated_scores`), the only
    scale on which an absolute threshold means anything. Without it a
    nonsense query returns a full page of 0.0001-scoring hits with no
    sign that they are junk, and the agent concludes the index is
    useless instead of trying an exact-name lookup.
    """
    if not results or not results.calibrated_scores:
        return None
    top = results[0].score
    if top >= WEAK_MATCH_SCORE:
        return None
    return (
        f"Note: no strong match — the best hit scores {top:.4f} of 1.0 "
        "(cross-encoder relevance), i.e. the index probably has nothing "
        "on this. If the query is an exact symbol name, call find_symbol; "
        "otherwise rephrase it in the words the code or docs would use."
    )


def _render(
    results: SearchResults,
    empty: str,
    note: str | None = None,
) -> str:
    if not results:
        return f"{note}\n\n{empty}" if note else empty
    body = "\n".join(result.render() for result in results)
    weak = _weak_match_hint(results)
    if weak:
        body = f"{weak}\n\n{body}"
    if results.has_more:
        body += (
            "\nNote: more matches exist beyond `limit`; increase `limit` "
            "or refine the query to see them."
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

    async def _search(
        call: Callable[[], Awaitable[SearchResults]],
        empty: str,
        repository: str | None,
    ) -> str:
        """Shared tail of every search_* tool: run ``call`` (the
        retrieval search), then render it with the repository's
        indexing note, the weak-match hint, and the "there really is
        more" hint (both carried on the results). ``call`` is
        invoked before ``app.indexing_note`` so an unknown-repository
        error from the search raises before indexing_note ever sees
        the (unvalidated) name."""
        return _render(await call(), empty, note=app.indexing_note(repository))

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_hdl(
        query: QueryArg,
        limit: LimitArg = DEFAULT_LIMIT,
        repository: RepositoryFilterArg = None,
        symbols: SymbolsArg = None,
        language: LanguageArg = None,
        mode: ModeArg = "hybrid",
    ) -> str:
        """Concept-level search of indexed HDL (VHDL, Verilog and
        SystemVerilog share one index: entities/modules, architectures,
        processes and always blocks, packages, functions, tasks).

        Use it when you do NOT know the name — "how is X done here",
        "find an example of Y" — or when the answer may be in a
        repository that is not in the working tree. If you DO know the
        identifier, call find_symbol instead: it is exact and typically
        two orders of magnitude cheaper, because each hit here returns a
        whole indexed construct. If you already know the file, call
        get_source, not this.

        Each hit quotes at most 40 lines of the matched chunk with a
        1-based line-number gutter, ending in the exact get_source call
        for anything elided; the navigation tools take 0-based lines, so
        pass N-1 for a line displayed as N. `score` is cross-encoder
        relevance in 0-1 when reranking is available (a top score below
        ~0.05 is reported as no real match), else a rank-fused score
        comparable only within one response."""
        return await _search(
            lambda: retrieval.search(
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
            repository,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_docs(
        query: QueryArg,
        limit: LimitArg = DEFAULT_LIMIT,
        repository: RepositoryFilterArg = None,
        symbols: SymbolsArg = None,
        mode: ModeArg = "hybrid",
    ) -> str:
        """Concept-level search of design documentation and coding
        standards, one result per section.

        Use it for "what is our convention for X", "what does the
        standard say about Y" — the configured coding-standards file is
        indexed here at high priority, and repository='coding-standards'
        restricts to it. Use search_hdl when the answer is RTL and
        search_knowledge when a requirement in the docs has to be
        followed into RTL and tests.

        Each hit quotes at most 40 lines of the matched chunk with a
        1-based line-number gutter, ending in the exact get_source call
        for anything elided; the navigation tools take 0-based lines, so
        pass N-1 for a line displayed as N. `score` is cross-encoder
        relevance in 0-1 when reranking is available (a top score below
        ~0.05 is reported as no real match), else a rank-fused score
        comparable only within one response."""
        return await _search(
            lambda: retrieval.search(
                CollectionName.DOCS,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No documentation results. Try a broader query, or check "
            "repository_status.",
            repository,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_code(
        query: QueryArg,
        limit: LimitArg = DEFAULT_LIMIT,
        repository: RepositoryFilterArg = None,
        symbols: SymbolsArg = None,
        mode: ModeArg = "hybrid",
    ) -> str:
        """Concept-level search of general, non-HDL source (C/C++,
        Python, ...), one result per function/class.

        Use it for the software side of a design — drivers, models,
        build and test scripts — or to follow an HDL identifier (a
        register name, a generic) into the software that drives it, via
        `symbols`. Use search_hdl for RTL, search_docs for specs, and
        plain grep/Read when the file is in the working tree and you
        want a literal string.

        Each hit quotes at most 40 lines of the matched chunk with a
        1-based line-number gutter, ending in the exact get_source call
        for anything elided; the navigation tools take 0-based lines, so
        pass N-1 for a line displayed as N. `score` is cross-encoder
        relevance in 0-1 when reranking is available (a top score below
        ~0.05 is reported as no real match), else a rank-fused score
        comparable only within one response."""
        return await _search(
            lambda: retrieval.search(
                CollectionName.CODE,
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No code results. Try a broader query, or check repository_status.",
            repository,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def search_knowledge(
        query: QueryArg,
        limit: KnowledgeLimitArg = KNOWLEDGE_LIMIT,
        repository: RepositoryFilterArg = None,
        symbols: SymbolsArg = None,
        mode: ModeArg = "hybrid",
    ) -> str:
        """Cross-domain search: HDL, documentation and general code at
        once, RRF-fused so the domains interleave fairly.

        Use it when the question spans them — a requirement stated in the
        docs, implemented in RTL and exercised by a testbench or C model
        — or when you cannot tell which domain holds the answer. When the
        domain is obvious, the single-domain tool is cheaper and
        sharper. `limit` defaults to 10 here (8 elsewhere).

        Each hit quotes at most 40 lines of the matched chunk with a
        1-based line-number gutter, ending in the exact get_source call
        for anything elided; the navigation tools take 0-based lines, so
        pass N-1 for a line displayed as N. `score` is cross-encoder
        relevance in 0-1 when reranking is available (a top score below
        ~0.05 is reported as no real match), else a rank-fused score
        comparable only within one response."""
        return await _search(
            lambda: retrieval.search_knowledge(
                query,
                limit,
                repository,
                tuple(symbols) if symbols else None,
                mode=mode,
            ),
            "No results in any domain. Try a broader query, or check "
            "repository_status.",
            repository,
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def get_source(
        repository: RepositoryArg,
        file: FileArg,
        start_line: Annotated[
            int | None,
            Field(
                description=(
                    "First line to return, 1-BASED and inclusive "
                    "(default: the start of the file). Use a search "
                    "result's or navigation hit's line range to read just "
                    "the construct instead of the whole file."
                )
            ),
        ] = None,
        end_line: Annotated[
            int | None,
            Field(
                description=(
                    "Last line to return, 1-BASED and inclusive "
                    "(default: the end of the file)."
                )
            ),
        ] = None,
    ) -> str:
        """Exact text of a known indexed file, or a line range of it, at
        the indexed commit and with repository/commit attribution.

        Use it whenever you already know the file — from a search
        result's source line, a navigation hit, or repository_files. Do
        not search for a file you can name: this is exact, returns only
        the lines asked for, and cannot drift from the index. If you do
        not know the path, call repository_files first rather than
        guessing one.

        Output carries the same 1-based line-number gutter search
        results use; call it with the start_line/end_line a search
        result's elision marker names to get the lines it did not
        quote."""
        return retrieval.get_source(repository, file, start_line, end_line)

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def repository_files(
        repository: RepositoryArg,
        pattern: Annotated[
            str | None,
            Field(
                description=(
                    "Glob matched against the repository-relative path, "
                    "where '*' crosses '/' — e.g. 'modules/counter/*' or "
                    "'*.vhd' (default: every indexed file). Narrow it "
                    "whenever you can: an unfiltered listing of a large "
                    "repository is mostly noise."
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum number of paths to list (default 200 — much "
                    "higher than the search tools', since a path costs "
                    "almost nothing). A truncation note is appended when "
                    "more match; refine `pattern` rather than raising it."
                )
            ),
        ] = 200,
    ) -> str:
        """List the indexed file paths of a repository — the candidate
        paths for get_source and the navigation tools.

        Use it whenever you are unsure of a path: it is far cheaper than
        searching for the file, and it removes the guessing that makes
        get_source fail. `pattern` narrows by glob."""
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
    async def find_definition(
        repository: RepositoryArg,
        file: FileArg,
        line: LineArg,
        character: CharacterArg,
    ) -> str:
        """Exact, compiler-backed go-to-definition for the symbol at a
        known position (vhdl_ls for VHDL, Veridian for
        Verilog/SystemVerilog).

        The cheapest accurate answer to "where is this declared" once you
        have a position — exact, and a fraction of a search. Know the
        name but not a position? Use find_symbol. Don't know the name at
        all? Use search_hdl. Positions are 0-based; results render as
        1-based `path:line:col` with source context. Cross-file
        resolution opens the repository's other same-language files
        (capped for responsiveness), so it usually works across files,
        but not always in a very large repository."""
        return await _find_definition(app, repository, file, line, character)

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def find_references(
        repository: RepositoryArg,
        file: FileArg,
        line: LineArg,
        character: CharacterArg,
        include_declaration: Annotated[
            bool,
            Field(
                description=(
                    "Whether the declaration site itself is listed among "
                    "the references (default true). Set it false when you "
                    "want only the uses — the declaration is filtered out "
                    "here even when the language server ignores the LSP "
                    "flag, as vhdl_ls does."
                )
            ),
        ] = True,
        limit: Annotated[
            int,
            Field(
                description=(
                    f"Maximum number of locations to render (default "
                    f"{REFERENCE_LIMIT}). Each one costs about five lines "
                    "of source context, so a common signal name would "
                    "otherwise return an unbounded response; a note says "
                    "how many were found when the list is truncated."
                )
            ),
        ] = REFERENCE_LIMIT,
    ) -> str:
        """Exact, compiler-backed find-references: every use site of the
        symbol at a known position (vhdl_ls for VHDL, Veridian for
        Verilog/SystemVerilog).

        Use it instead of grepping a name: it is scope-aware, so it will
        not match a same-named signal in another entity, and it costs a
        fraction of a search. Use search_hdl with `symbols` only for a
        fuzzy sweep that should also reach docs and testbenches.
        Positions are 0-based; results render as 1-based
        `path:line:col` with source context, capped at `limit`."""
        return await _find_references(
            app, repository, file, line, character, include_declaration, limit
        )

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def hover_info(
        repository: RepositoryArg,
        file: FileArg,
        line: LineArg,
        character: CharacterArg,
    ) -> str:
        """Exact, compiler-backed hover: the analyzer's own
        declaration/type/doc-comment text for the symbol at a known
        position, as an IDE would show it (vhdl_ls for VHDL, Veridian
        for Verilog/SystemVerilog).

        The cheapest way to answer "what type, width or signature does
        this have" without reading the file at all. It needs an exact
        position: use find_symbol to turn a name into one, or search_hdl
        when the name itself is unknown. Positions are 0-based."""
        return await _hover_info(app, repository, file, line, character)

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def find_symbol(
        query: Annotated[
            str,
            Field(
                description=(
                    "The identifier to look up, e.g. 'axi_lite_pkg' or "
                    "'fifo_wr_ptr'. Matched by the language server's own "
                    "name matching (substring/fuzzy on the NAME), never "
                    "semantically — a natural-language phrase belongs in "
                    "search_hdl instead."
                )
            ),
        ],
        repository: RepositoryFilterArg = None,
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum number of declarations to return (default "
                    "8). Repositories are searched in configured order "
                    "until this many hits are collected."
                )
            ),
        ] = DEFAULT_LIMIT,
    ) -> str:
        """Exact, compiler-backed workspace symbol lookup: where a named
        entity/module/package/function/signal is declared (vhdl_ls for
        VHDL, Veridian for Verilog/SystemVerilog).

        Try this FIRST whenever you know the identifier. It is the
        cheapest tool here and it is exact, while search_hdl answers the
        same question with whole constructs, for one to two orders of
        magnitude more tokens and less precisely. Matching is on the
        name, not the meaning: for a concept or a natural-language
        question use search_hdl. Results render as 1-based
        `path:line:col` with source context — feed one straight into
        find_references or hover_info."""
        return await _find_symbol(app, repository, query, limit)

    @mcp.tool(annotations=_READ_ONLY)
    @_handle_errors
    async def repository_status() -> str:
        """What is indexed, what is syncing right now, what failed, and
        the exact repository names every other tool expects.

        Call it when you need a repository name (in zero-config mode the
        name carries a hash suffix, e.g. 'vhdl-ai-test-582e8509' — never
        guess it), when a search comes back thin or empty (a sync may
        still be running, and the 'sync:' line says so), or when
        navigation reports an analyzer problem. Reports per repository:
        sync state (in progress / failed / pending / idle), last indexed
        commit, chunk and file counts, last sync time and last error;
        then the HDL analyzers (vhdl_ls, Veridian) with availability,
        version, and whether semantic (lsp) or fallback parsing is in
        effect. A configured coding-standards file appears as the
        'coding-standards' pseudo-repository (content hash in place of a
        commit)."""
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
            sync_state = app.sync_state(
                status.name, status.indexed_commit, status.last_sync_error
            )
            lines.append(
                f"- {status.name} ({source}, priority {status.priority}, "
                f"domains: {domains})\n"
                f"  sync: {sync_state}\n"
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
            standards_state = app.sync_state(
                CODING_STANDARDS_REPO, state.indexed_commit, state.last_sync_error
            )
            standards_line = (
                f"- {CODING_STANDARDS_REPO} (file {app.config.coding_standards}, "
                f"priority {app.config.coding_standards_priority})\n"
                f"  sync: {standards_state}\n"
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
        for analyzer in app.analyzer_statuses().values():
            lines.append(f"- {analyzer.name}: {analyzer.describe()}")
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
    async def sync_repositories(
        repositories: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Repository names to sync (default: every configured "
                    "repository, which is slower). An unknown name is "
                    "rejected up front."
                )
            ),
        ] = None,
    ) -> str:
        """Incrementally sync repositories: fetch the ref, chunk changed
        files, update the index.

        You normally never need this — the index syncs itself on a timer,
        and local working repositories within seconds of an edit. Use it
        when repository_status shows a sync failed and you have fixed the
        cause, or when a just-made change must be indexed immediately.
        Safe to call any time; failures are contained per repository and
        reported."""
        reports = await app.sync_all(repositories)
        return _render_report(reports)

    @mcp.tool(annotations=_READ_WRITE)
    @_handle_errors
    async def reindex_repository(
        repository: Annotated[
            str,
            Field(
                description=(
                    "The repository to rebuild from scratch, by its exact "
                    "name from repository_status ('coding-standards' for "
                    "the configured standards file)."
                )
            ),
        ],
    ) -> str:
        """Fully reindex one repository: drop and rebuild all of its
        chunks.

        Much more expensive than sync_repositories, and rarely the right
        tool: use it after a config change that alters what gets indexed,
        or to repair an index you have reason to believe has drifted."""
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


async def _cancel_and_wait(*tasks: asyncio.Task[Any] | None) -> None:
    """Cancel every task (``None`` entries are skipped), then await each
    in turn, suppressing the resulting ``CancelledError``.

    Cancelling all of them up front before awaiting any preserves the
    same teardown order as three separate cancel/await blocks written
    out by hand: every task's cancellation is requested before we start
    waiting on any one of them, instead of staggering it while an
    earlier task's await is still pending.
    """
    for task in tasks:
        if task is not None:
            task.cancel()
    for task in tasks:
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


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
        await _cancel_and_wait(initial_sync, sync_task, poll_task)
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
            "($CORVIDEX_MCP_CONFIG or the project-local .corvidex) and "
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
            "config file (default: $CORVIDEX_MCP_CONFIG, else the "
            "project-local .corvidex in the project directory: "
            "$CORVIDEX_MCP_PROJECT_DIR, else the current directory). "
            "Running without any config file is supported and indexes "
            "the project directory"
        ),
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help=(
            "write a commented .corvidex template to the project directory "
            "and exit (never overwrites an existing file); no config file is "
            "needed to run"
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
        "--vhdl-ls-libraries-dir",
        default=None,
        metavar="PATH",
        help=(
            "override vhdl_ls_libraries_dir: the VHDL standard-library "
            "sources vhdl_ls needs. A property of the vhdl_ls install rather "
            "than of any one project (a 'cargo install'ed binary ships "
            "without them and panics on every invocation), so it usually "
            "belongs on the launcher command line, where it applies to every "
            "workspace, rather than in a per-project .corvidex"
        ),
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

    The config file is selected by ``--config``, else ``CORVIDEX_MCP_CONFIG``,
    else the project-local ``.corvidex`` in the current directory (see
    :func:`corvidex_mcp.config.load_config`); ``--data-dir``/
    ``--sync-interval``/``--vhdl-ls-path``/``--vhdl-ls-libraries-dir``/
    ``--veridian-path``/``--log-level``/``--num-threads`` override the
    file's values.
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
    "vhdl_ls_libraries_dir",
    "veridian_path",
    "log_level",
)


def _init_config() -> int:
    """Write the commented ``.corvidex`` template to the project directory
    for ``--init-config``, refusing to overwrite an existing file."""
    path = project_config_path()
    if path.exists():
        print(f"corvidex-mcp: {path} already exists, leaving it alone")
        return 1
    path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
    print(f"corvidex-mcp: wrote {path}")
    return 0


def _own_source_tree() -> Path | None:
    """The ``corvidex-mcp`` checkout this server is running from, or ``None``
    when it runs from an installed wheel rather than a source tree."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "src" / "corvidex_mcp").is_dir() else None


def _warn_if_indexing_itself(config: AppConfig) -> None:
    """Warn when zero-config mode picked corvidex-mcp's own source tree.

    Indexing corvidex-mcp itself is legitimate when its maintainer means
    it, but far more often it means the launcher started the server in the
    wrong directory and the user's actual code is not being indexed at
    all — a silent, confusing failure worth one loud line.
    """
    own = _own_source_tree()
    if own is None or not any(
        repo.auto_indexed and repo.path == own for repo in config.repositories
    ):
        return
    logger.warning(
        "auto-indexing corvidex-mcp's own source tree (%s) because that is "
        "the working directory. If this is not what you meant, the launcher "
        "is starting the server in the wrong place: use 'uv --project DIR "
        "run', not 'uv --directory DIR run' (which changes the working "
        "directory), or set %s to your workspace root.",
        own,
        PROJECT_DIR_ENV,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server over stdio (the ``corvidex-mcp`` entry point)."""
    if _parse_args(argv).init_config:
        raise SystemExit(_init_config())
    try:
        config = config_from_args(argv)
    except (ConfigError, ValidationError) as exc:
        print(f"corvidex-mcp: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    setup_logging(config.log_level, config.log_file)
    _warn_if_indexing_itself(config)
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
