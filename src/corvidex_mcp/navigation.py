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

#: Default cap on the locations ``find_references`` renders. Every
#: location costs a header plus ``2 * CONTEXT_LINES + 1`` lines of
#: source, so an uncapped call on a common name (``clk``, ``rst_n``,
#: ``valid``) returns an unbounded response; the reply says how many
#: were found when it truncates, the way the search tools do.
REFERENCE_LIMIT = 20

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


def _validate_position(file: str, lines: list[str], line: int, character: int) -> None:
    """Reject a position that cannot name anything in ``file``.

    An out-of-range or negative position used to reach the language
    server and come back as the same flat "No definition found at that
    position." as a genuinely undefined symbol, so a caller could not
    tell a typo'd position from a missing declaration and would
    conclude the symbol does not exist. Positions are 0-based (LSP
    convention); the messages say so and name the file's real extent,
    so the caller can correct the position without guessing.
    """
    if line < 0 or character < 0:
        raise RetrievalError(
            f"invalid position {line}:{character} in {file!r}: line and "
            "character are 0-based and must not be negative (the very "
            "first character of a file is line 0, character 0)"
        )
    if not lines:
        raise RetrievalError(
            f"{file!r} is empty, so no position in it can name a symbol"
        )
    if line >= len(lines):
        raise RetrievalError(
            f"line {line} is past the end of {file!r}: the file has "
            f"{len(lines)} lines, so the last addressable line is "
            f"{len(lines) - 1} (positions here are 0-based — line "
            f"{len(lines) - 1} is what an editor shows as line "
            f"{len(lines)})"
        )
    width = len(lines[line])
    if character > width:
        raise RetrievalError(
            f"character {character} is past the end of line {line} in "
            f"{file!r}: that line is {width} characters long, so the last "
            f"addressable character is {max(width - 1, 0)} (positions here "
            "are 0-based)"
        )


#: A VHDL selected name — ``entity``, ``cnn_accel.cnn_accel_pkg``,
#: ``work.foo.bar`` — used only to explain an unresolved position, so it
#: deliberately ignores extended identifiers and whitespace around dots.
_SELECTED_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")


def _selected_name_at(lines: list[str], line: int, character: int) -> str:
    """The dotted VHDL name covering ``character``, or '' when the
    position is not on an identifier at all."""
    if not 0 <= line < len(lines):
        return ""
    text = lines[line]
    for match in _SELECTED_NAME.finditer(text):
        if match.start() <= character < match.end():
            return match.group(0)
    return ""


def _no_result_message(what: str, lines: list[str], line: int, character: int) -> str:
    """ "No <what> found" plus what can be told about *why*, cheaply.

    The flat message alone is indistinguishable between "the symbol is
    not declared anywhere", "you are pointing at whitespace" and "the
    library in this library-qualified name is not configured" — and a
    caller reading it concludes the symbol does not exist. The last
    case is the common one in tsfpga/hdl-modules projects, where
    ``<library>.<entity>`` instantiation is the standard style, so it
    is called out explicitly.
    """
    base = f"No {what} found at that position."
    name = _selected_name_at(lines, line, character)
    if not name:
        text = lines[line] if 0 <= line < len(lines) else ""
        if not text.strip():
            return (
                f"{base} Line {line} is blank (positions here are 0-based, "
                f"so this is line {line + 1} in an editor) — point at the "
                "identifier itself."
            )
        return (
            f"{base} Character {character} of line {line} is not part of an "
            f"identifier (the line is {text.strip()[:80]!r}) — positions "
            "here are 0-based and must sit on the name itself."
        )
    if "." in name:
        library, _, suffix = name.partition(".")
        if library.lower() == "work":
            return (
                f"{base} {name!r} names {suffix!r} in the file's own library "
                "('work'), so either it is not declared there or the "
                "declaring file is not part of this analysis session."
            )
        return (
            f"{base} {name!r} is library-qualified and the library "
            f"{library!r} did not resolve. vhdl_ls can only resolve "
            "'<library>.<name>' when that library is declared in the "
            "workspace's vhdl_ls.toml: check the repository's own "
            "vhdl_ls.toml (or its vhdl_ls_hook) if it has one — otherwise "
            "the built-in generated config infers libraries from the "
            "'modules/<library>/...' layout, and this file does not match "
            "it. The symbol may well exist; this is a configuration "
            "result, not a 'not declared' one."
        )
    return (
        f"{base} The position is on {name!r}; if that symbol is declared in "
        "another file, that file may be outside this session's file set "
        f"(capped at {MAX_SESSION_FILES} same-language files) or in a "
        "library the vhdl_ls configuration does not declare."
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
    position: tuple[int, int],
) -> tuple[T, LspClient, Path, list[str]]:
    """Shared session lifecycle for definition/references/hover.

    Resolves the repository, picks the analyzer from ``file``'s
    extension, validates ``position`` against the file's real extent
    *before* paying for a language-server session, opens a
    same-language file set (see :func:`_session_files`), runs ``call``,
    and returns its result alongside the (now shut down, but still
    readable) client — for ``supports_*`` checks — the checkout dir,
    for path rendering, and the file's lines, for explaining an empty
    result.
    """
    cfg = _resolve_repository(app, repository)
    analyzer = _analyzer_for(file)
    _check_file_exists(app, cfg, file)
    try:
        source = app.git.read_file(cfg, file)
    except GitError as exc:
        raise RetrievalError(str(exc)) from exc
    lines = source.splitlines()
    _validate_position(file, lines, *position)
    files = _session_files(app, cfg, file, _EXTENSIONS[analyzer])
    async with _session(app, cfg, analyzer, files) as (lsp, repo_dir):
        result = await call(lsp, repo_dir / file)
    return result, lsp, repo_dir, lines


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
    locations, lsp, repo_dir, lines = await _navigate(
        app,
        repository,
        file,
        lambda c, path: c.definition(path, line, character),
        (line, character),
    )
    if not lsp.supports_definition:
        return (
            "The language server for this file does not advertise "
            "go-to-definition support."
        )
    if not locations:
        return _no_result_message("definition", lines, line, character)
    return _format_locations(app, repository, repo_dir, locations)


def _without_declaration(
    locations: tuple[Location, ...], declarations: tuple[Location, ...]
) -> tuple[Location, ...]:
    """``locations`` minus the declaration site(s).

    ``include_declaration=False`` is sent to the server as the LSP
    ``context.includeDeclaration`` flag, but vhdl_ls ignores it and
    returns the declaration anyway (verified: the two responses are
    byte-identical), so the flag has to be honoured here as well. The
    declaration is identified by asking the same session for
    go-to-definition at the same position and dropping every reference
    on a declaration's file and line — matching on the line rather than
    the exact column because a server may report the declaration's name
    range and its reference range with different start columns.
    """
    if not declarations:
        return locations
    declared = {(_uri_to_path(loc.uri), loc.start_line) for loc in declarations}
    return tuple(
        loc
        for loc in locations
        if (_uri_to_path(loc.uri), loc.start_line) not in declared
    )


async def find_references(
    app: VhdlRagApp,
    repository: str,
    file: str,
    line: int,
    character: int,
    include_declaration: bool = True,
    limit: int = REFERENCE_LIMIT,
) -> str:
    """Exact find-references (vhdl_ls/Veridian), not similarity search.

    ``line``/``character`` are 0-based (LSP convention); results are
    rendered as 1-based ``path:line:col``, capped at ``limit`` with a
    note naming the true total (see :data:`REFERENCE_LIMIT`). See the
    module docstring for the cross-file resolution caveat.
    """
    if limit < 1:
        raise RetrievalError("limit must be at least 1")

    async def call(
        c: LspClient, path: Path
    ) -> tuple[
        tuple[Location, ...],
        tuple[Location, ...],
    ]:
        found = await c.references(
            path, line, character, include_declaration=include_declaration
        )
        if include_declaration:
            return found, ()
        # Same session, same position: what the server calls the
        # declaration, so it can be filtered out of the references it
        # returned despite includeDeclaration=false.
        return found, await c.definition(path, line, character)

    (locations, declarations), lsp, repo_dir, lines = await _navigate(
        app, repository, file, call, (line, character)
    )
    if not lsp.supports_references:
        return (
            "The language server for this file does not advertise "
            "find-references support."
        )
    if not include_declaration:
        locations = _without_declaration(locations, declarations)
    if not locations:
        return _no_result_message("references", lines, line, character)
    shown = locations[:limit]
    body = _format_locations(app, repository, repo_dir, shown)
    if len(locations) > limit:
        body += (
            f"\n\nNote: {len(locations)} references found; showing the "
            f"first {limit}. Increase `limit` to see the rest."
        )
    return body


async def hover_info(
    app: VhdlRagApp, repository: str, file: str, line: int, character: int
) -> str:
    """Exact hover text (vhdl_ls/Veridian: declaration/type/doc comment).

    ``line``/``character`` are 0-based (LSP convention). Not a
    substitute for search_hdl when the exact position is unknown.
    """
    text, lsp, _repo_dir, lines = await _navigate(
        app,
        repository,
        file,
        lambda c, path: c.hover(path, line, character),
        (line, character),
    )
    if not lsp.supports_hover:
        return "The language server for this file does not advertise hover support."
    if text is None:
        return _no_result_message("hover information", lines, line, character)
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
    if repository is not None:
        # Up front, like every other tool: the per-repository loop below
        # swallows RetrievalError to keep one broken repository from
        # failing a multi-repository search, which would otherwise turn
        # a misspelled name into a plausible-looking "No symbols
        # matching ..." instead of an error.
        _resolve_repository(app, repository)
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
