"""Exact, LSP-backed semantic navigation: definition, references, hover,
workspace symbol search.

This complements :mod:`corvidex_mcp.retrieval`'s fuzzy/semantic search
tools (``search_hdl`` and friends), which rank chunks by embedding
similarity plus full-text match. The tools here instead ask the same
language servers the indexing pipeline uses for chunking — vhdl_ls for
VHDL, Veridian for Verilog/SystemVerilog (see
:mod:`corvidex_mcp.lsp.client`) — for exact, compiler-backed answers:
"where is this symbol declared", "who references it", "what is its
signature/doc comment", "which symbols named roughly this exist in the
workspace". Use these when the caller already knows (or has from a
search result) a precise file:line:character and wants the analyzer's
own answer, not a similarity ranking.

Each call here starts a short-lived, throwaway LSP session scoped to
one repository: unlike the indexing pipeline's sessions (one per sync,
reused across every changed file, see
:mod:`corvidex_mcp.indexing.pipeline`), a navigation session is opened
fresh for the one tool call and shut down before it returns. To give
cross-file resolution (an entity used in one file, declared in
another) a real chance to work, every other same-language file the
index already knows about for the repository is opened too — capped
at :data:`MAX_SESSION_FILES` so one call cannot turn into a full
resync. A file outside that cap, or belonging to a language the
repository's ``domains``/``exclude`` settings do not index, is
invisible to the analyzer for that call.

Every position taken or returned here is 0-based (line and character),
matching the LSP wire format; results are rendered 1-based
(``path:line:col``) the way a human reads a file — the docstrings
below call this out at each boundary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from .config import ConfigError, RepositoryConfig
from .git_manager import GitError
from .lsp import (
    Location,
    LspClient,
    VeridianLsp,
    VhdlLsp,
    WorkspaceSymbolInfo,
    default_libraries_dir,
)
from .models import CollectionName
from .retrieval import RetrievalError
from .routing import SYSTEMVERILOG_EXTENSIONS, VERILOG_EXTENSIONS, VHDL_EXTENSIONS

if TYPE_CHECKING:
    from .server import VhdlRagApp

logger = logging.getLogger(__name__)

#: Cap on the files opened in one navigation session for cross-file
#: resolution. Opening an entire large repository on every tool call
#: would make each call as slow as a full sync; the queried file is
#: always included even when this cap would otherwise exclude it.
MAX_SESSION_FILES = 200

#: Lines of source context shown before/after each hit.
CONTEXT_LINES = 2

#: Extensions handled by each analyzer (see :mod:`corvidex_mcp.routing`).
_EXTENSIONS: dict[str, frozenset[str]] = {
    "vhdl_ls": VHDL_EXTENSIONS,
    "veridian": VERILOG_EXTENSIONS | SYSTEMVERILOG_EXTENSIONS,
}

#: Common LSP SymbolKind values, for a readable find_symbol rendering
#: (see the LSP spec's SymbolKind enum; unlisted kinds fall back to
#: "kind <n>").
_SYMBOL_KIND_NAMES: dict[int, str] = {
    2: "module",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    23: "struct",
    26: "type parameter",
}


def _resolve_repository(app: VhdlRagApp, name: str) -> RepositoryConfig:
    """Same resolution as ``VhdlRagApp._config_or_error`` (a repository
    name to its config), reimplemented here to avoid importing
    :mod:`corvidex_mcp.server` (which imports these tools)."""
    try:
        return app.config.repository(name)
    except ConfigError as exc:
        raise RetrievalError(str(exc)) from exc


def _analyzer_for(file: str) -> str:
    """``'vhdl_ls'`` or ``'veridian'`` for one file's extension.

    Raises RetrievalError for anything else: exact navigation only
    covers HDL source (the same languages the indexing pipeline feeds
    to these analyzers).
    """
    ext = Path(file).suffix.lower()
    if ext in VHDL_EXTENSIONS:
        return "vhdl_ls"
    if ext in VERILOG_EXTENSIONS or ext in SYSTEMVERILOG_EXTENSIONS:
        return "veridian"
    raise RetrievalError(
        f"{file!r} is not a VHDL/Verilog/SystemVerilog file; exact "
        "navigation only supports HDL source (use search_hdl or "
        "search_code for other file types)"
    )


def _check_file_exists(app: VhdlRagApp, cfg: RepositoryConfig, file: str) -> None:
    if ".." in Path(file).parts or file.startswith("/"):
        raise RetrievalError(f"invalid repository file path: {file!r}")
    path = app.git.repo_dir(cfg) / file
    if not path.is_file():
        raise RetrievalError(
            f"file {file!r} not found in {cfg.name!r}; use repository_files "
            "to list valid paths in this repository"
        )


def _session_files(
    app: VhdlRagApp, cfg: RepositoryConfig, file: str, extensions: frozenset[str]
) -> list[str]:
    """Same-language indexed files for a session, ``file`` always included."""
    files, _truncated = app.retrieval.list_files(cfg.name)
    same_language = [f for f in files if Path(f).suffix.lower() in extensions]
    if file not in same_language:
        same_language.append(file)
    if len(same_language) <= MAX_SESSION_FILES:
        return same_language
    logger.info(
        "%s: %d same-language file(s); capping the navigation session to "
        "%d for responsiveness (cross-file resolution may miss symbols "
        "defined outside the capped set)",
        cfg.name,
        len(same_language),
        MAX_SESSION_FILES,
    )
    rest = [f for f in same_language if f != file][: MAX_SESSION_FILES - 1]
    return [file, *rest]


@asynccontextmanager
async def _session(
    app: VhdlRagApp, cfg: RepositoryConfig, analyzer: str, files: list[str]
) -> AsyncIterator[tuple[LspClient, Path]]:
    """Start a short-lived LSP session over ``files`` in ``cfg``'s
    checkout, yield ``(lsp, repo_dir)``, then always shut it down."""
    status = app.analyzer_statuses()[analyzer]
    if not status.available:
        raise RetrievalError(
            f"{analyzer} is not available ({status.error}); exact "
            "navigation needs the same language server the indexing "
            "pipeline uses (see repository_status)"
        )
    assert status.path is not None
    repo_dir = app.git.repo_dir(cfg)
    if not repo_dir.is_dir():
        raise RetrievalError(
            f"repository {cfg.name!r} has no checkout yet; call sync_repositories first"
        )
    contents: dict[str, str] = {}
    for f in files:
        try:
            contents[f] = app.git.read_file(cfg, f)
        except GitError as exc:
            logger.warning("navigation: skipping unreadable file %s: %s", f, exc)
    lsp: LspClient
    if analyzer == "vhdl_ls":
        libraries_dir = app.config.vhdl_ls_libraries_dir or default_libraries_dir(
            status.path
        )
        lsp = VhdlLsp(
            status.path,
            repo_dir,
            libraries_dir=libraries_dir,
            vhdl_ls_hook=cfg.vhdl_ls_hook,
            files=tuple(contents),
        )
    else:
        lsp = VeridianLsp(status.path, repo_dir, config_hook=cfg.veridian_hook)
    try:
        await lsp.start()
        for f, text in contents.items():
            await lsp.open_document(repo_dir / f, text=text)
        await lsp.wait_until_quiet(timeout=max(20.0, 2.0 * len(contents)))
        yield lsp, repo_dir
    finally:
        await lsp.shutdown()


async def _navigate[T](
    app: VhdlRagApp,
    repository: str,
    file: str,
    call: Callable[[LspClient, Path], Awaitable[T]],
) -> tuple[T, LspClient, Path]:
    """Shared session lifecycle for definition/references/hover.

    Resolves the repository, picks the analyzer from ``file``'s
    extension, opens a same-language file set (see
    :func:`_session_files`), runs ``call``, and returns its result
    alongside the (now shut down, but still readable) client — for
    ``supports_*`` checks — and the checkout dir, for path rendering.
    """
    cfg = _resolve_repository(app, repository)
    analyzer = _analyzer_for(file)
    _check_file_exists(app, cfg, file)
    files = _session_files(app, cfg, file, _EXTENSIONS[analyzer])
    async with _session(app, cfg, analyzer, files) as (lsp, repo_dir):
        result = await call(lsp, repo_dir / file)
    return result, lsp, repo_dir


#: A URI path component shaped like ``/C:/...`` — the POSIX-style
#: leading slash ``Path.as_uri()``/LSP servers put in front of a
#: Windows drive letter. No genuine POSIX path is ever shaped this
#: way, so stripping it is safe on every platform (this makes the
#: fix testable on non-Windows CI too, rather than needing an
#: ``os.name`` check that only exercises on Windows runners).
_WINDOWS_DRIVE_URI_PATH = re.compile(r"^/([A-Za-z]:)(/.*)?$")


def _uri_to_path(uri: str) -> Path:
    raw = unquote(urlparse(uri).path)
    match = _WINDOWS_DRIVE_URI_PATH.match(raw)
    if match:
        raw = match.group(1) + (match.group(2) or "")
    return Path(raw)


def _relative_path(path: Path, repo_dir: Path) -> str:
    try:
        return path.relative_to(repo_dir).as_posix()
    except ValueError:
        return str(path)


def _format_hit(
    app: VhdlRagApp, repository: str, repo_dir: Path, uri: str, line: int, char: int
) -> str:
    """One ``path:line:col`` header plus a source snippet around it.

    ``line``/``char`` are the 0-based LSP position; the header renders
    1-based line/column, matching how editors and compilers report
    positions to humans.
    """
    rel = _relative_path(_uri_to_path(uri), repo_dir)
    line_1based = line + 1
    col_1based = char + 1
    header = f"{repository}:{rel}:{line_1based}:{col_1based}"
    start = max(1, line_1based - CONTEXT_LINES)
    end = line_1based + CONTEXT_LINES
    try:
        snippet = app.retrieval.get_source(repository, rel, start, end)
    except RetrievalError:
        snippet = f"(source unavailable for {rel})"
    return f"{header}\n{snippet}"


def _format_locations(
    app: VhdlRagApp, repository: str, repo_dir: Path, locations: tuple[Location, ...]
) -> str:
    return "\n\n".join(
        _format_hit(app, repository, repo_dir, loc.uri, loc.start_line, loc.start_char)
        for loc in locations
    )


# -- public navigation API ----------------------------------------------------


async def find_definition(
    app: VhdlRagApp, repository: str, file: str, line: int, character: int
) -> str:
    """Exact go-to-definition (vhdl_ls/Veridian), not similarity search.

    ``line``/``character`` are 0-based (LSP convention); results are
    rendered as 1-based ``path:line:col``. See the module docstring
    for the cross-file resolution caveat (same-language files up to
    :data:`MAX_SESSION_FILES` are opened alongside ``file``).
    """
    locations, lsp, repo_dir = await _navigate(
        app, repository, file, lambda c, path: c.definition(path, line, character)
    )
    if not lsp.supports_definition:
        return (
            "The language server for this file does not advertise "
            "go-to-definition support."
        )
    if not locations:
        return "No definition found at that position."
    return _format_locations(app, repository, repo_dir, locations)


async def find_references(
    app: VhdlRagApp,
    repository: str,
    file: str,
    line: int,
    character: int,
    include_declaration: bool = True,
) -> str:
    """Exact find-references (vhdl_ls/Veridian), not similarity search.

    ``line``/``character`` are 0-based (LSP convention); results are
    rendered as 1-based ``path:line:col``. See the module docstring
    for the cross-file resolution caveat.
    """
    locations, lsp, repo_dir = await _navigate(
        app,
        repository,
        file,
        lambda c, path: c.references(
            path, line, character, include_declaration=include_declaration
        ),
    )
    if not lsp.supports_references:
        return (
            "The language server for this file does not advertise "
            "find-references support."
        )
    if not locations:
        return "No references found at that position."
    return _format_locations(app, repository, repo_dir, locations)


async def hover_info(
    app: VhdlRagApp, repository: str, file: str, line: int, character: int
) -> str:
    """Exact hover text (vhdl_ls/Veridian: declaration/type/doc comment).

    ``line``/``character`` are 0-based (LSP convention). Not a
    substitute for search_hdl when the exact position is unknown.
    """
    text, lsp, _repo_dir = await _navigate(
        app, repository, file, lambda c, path: c.hover(path, line, character)
    )
    if not lsp.supports_hover:
        return "The language server for this file does not advertise hover support."
    if text is None:
        return "No hover information at that position."
    return text


def _hdl_repository_names(app: VhdlRagApp) -> list[str]:
    return [
        repo.name
        for repo in app.config.repositories
        if CollectionName.HDL in repo.domains
    ]


async def find_symbol(
    app: VhdlRagApp, repository: str | None, query: str, limit: int = 8
) -> str:
    """Exact workspace symbol search (vhdl_ls/Veridian), not similarity
    search — matches on (sub)string/fuzzy-name matching *implemented by
    the language server itself*, not embeddings. Use search_hdl instead
    for conceptual/natural-language queries. ``repository`` restricts
    the search to one repository; omit it to search every configured
    HDL repository (in configured order) until ``limit`` results are
    collected.
    """
    query = query.strip()
    if not query:
        raise RetrievalError("query must not be empty")
    repo_names = [repository] if repository is not None else _hdl_repository_names(app)
    if not repo_names:
        return "No HDL repositories configured."
    hits: list[tuple[str, Path, WorkspaceSymbolInfo]] = []
    for name in repo_names:
        if len(hits) >= limit:
            break
        try:
            cfg = _resolve_repository(app, name)
        except RetrievalError:
            continue
        files, _truncated = app.retrieval.list_files(name)
        for analyzer, extensions in _EXTENSIONS.items():
            if len(hits) >= limit:
                break
            session_files = [f for f in files if Path(f).suffix.lower() in extensions]
            if not session_files:
                continue
            status = app.analyzer_statuses()[analyzer]
            if not status.available:
                continue
            try:
                async with _session(
                    app, cfg, analyzer, session_files[:MAX_SESSION_FILES]
                ) as (lsp, repo_dir):
                    symbols = await lsp.workspace_symbol(query)
            except RetrievalError as exc:
                logger.warning("find_symbol: %s/%s failed: %s", name, analyzer, exc)
                continue
            for symbol in symbols:
                hits.append((name, repo_dir, symbol))
                if len(hits) >= limit:
                    break
    if not hits:
        return (
            f"No symbols matching {query!r}. This is exact workspace symbol "
            "search (vhdl_ls/Veridian), not fuzzy — try search_hdl for a "
            "conceptual/natural-language query instead."
        )
    lines: list[str] = []
    for name, repo_dir, symbol in hits[:limit]:
        kind_name = _SYMBOL_KIND_NAMES.get(symbol.kind, f"kind {symbol.kind}")
        header = _format_hit(
            app, name, repo_dir, symbol.uri, symbol.start_line, symbol.start_char
        )
        lines.append(f"{kind_name} {symbol.name}\n{header}")
    return "\n\n".join(lines)
