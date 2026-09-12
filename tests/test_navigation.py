"""Tests for :mod:`corvidex_mcp.navigation` (exact LSP-backed navigation).

Runs fully offline: local file:// git remotes, fake embedding
providers, and one fake LSP server binary (speaking the same framing
as vhdl_ls/Veridian, used for both since the protocol layer does not
care which analyzer it stands in for) that also answers
definition/references/hover/workspace_symbol so the navigation tools
can be exercised end to end without a real language server.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from capability import sqlite_extensions_supported
from fake_lsp_util import executable_lsp_script

from corvidex_mcp.config import AppConfig, RepositoryConfig
from corvidex_mcp.embeddings.provider import FastEmbedProvider
from corvidex_mcp.embeddings.providers import EmbeddingProviders
from corvidex_mcp.git_manager import GitManager
from corvidex_mcp.indexing.pipeline import IndexPipeline
from corvidex_mcp.lsp import build_analyzer_statuses
from corvidex_mcp.navigation import (
    REFERENCE_LIMIT,
    _no_result_message,
    _selected_name_at,
    _uri_to_path,
    _validate_position,
    find_definition,
    find_references,
    find_symbol,
    hover_info,
)
from corvidex_mcp.retrieval import RetrievalError, RetrievalService
from corvidex_mcp.state import StateStore
from corvidex_mcp.vector_store import VectorStore

pytestmark = pytest.mark.skipif(
    not sqlite_extensions_supported(),
    reason=(
        "stdlib SQLite lacks loadable-extension support (the sqlite-vec "
        "extension cannot load; use CPython 3.14 or a system/homebrew "
        "Python)"
    ),
)

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "protocol.file.allow",
    "GIT_CONFIG_VALUE_0": "always",
}

# A fake server that stands in for both vhdl_ls and Veridian: it always
# reports every file clean (no diagnostics) and returns None for
# documentSymbol (so the indexing pipeline's structural fallback parses
# the real fixture content), but fully answers the navigation methods:
#
# - textDocument/definition: the response shape/target is selected by
#   the request's `character` (0: null, 1: single Location in the same
#   file, 2: a Location[] pointing at a *different* file in the repo —
#   exercising cross-file rendering, 3+: a LocationLink[]).
# - textDocument/references: one hit, plus the declaration when
#   `includeDeclaration` is true.
# - textDocument/hover: null/string/MarkupContent by `character`.
# - workspace/symbol: one hit unless the query is "nomatch".
FAKE_NAV_LSP = r"""#!/usr/bin/env python3
import json
import sys

if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
    print("fake-nav-lsp 1.0.0")
    sys.exit(0)


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get(b"content-length", b"0"))
    return json.loads(sys.stdin.buffer.read(length))


def send(obj):
    body = json.dumps(obj).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    sys.stdout.buffer.write(frame + body)
    sys.stdout.buffer.flush()


def pos_range(line, char):
    return {
        "start": {"line": line, "character": char},
        "end": {"line": line, "character": char + 3},
    }


read_message()  # initialize
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "capabilities": {
                "documentSymbolProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
                "hoverProvider": True,
                "workspaceSymbolProvider": True,
            }
        },
    }
)
msg = read_message()  # initialized
assert msg is not None and msg.get("method") == "initialized", msg
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": []},
            }
        )
    elif method == "textDocument/documentSymbol":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "textDocument/definition":
        char = msg["params"]["position"]["character"]
        uri = msg["params"]["textDocument"]["uri"]
        if char == 0:
            result = None
        elif char == 1:
            result = {"uri": uri, "range": pos_range(2, 0)}
        elif char == 2:
            other = uri.rsplit("/", 1)[0] + "/other.vhd"
            result = [{"uri": other, "range": pos_range(0, 0)}]
        else:
            result = [
                {
                    "targetUri": uri,
                    "targetRange": pos_range(4, 0),
                    "targetSelectionRange": pos_range(4, 2),
                }
            ]
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method == "textDocument/references":
        uri = msg["params"]["textDocument"]["uri"]
        include_decl = msg["params"]["context"]["includeDeclaration"]
        result = [{"uri": uri, "range": pos_range(3, 0)}]
        if include_decl:
            result.append({"uri": uri, "range": pos_range(0, 0)})
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method == "textDocument/hover":
        char = msg["params"]["position"]["character"]
        if char == 0:
            result = None
        elif char == 1:
            result = {"contents": "plain hover text"}
        else:
            result = {"contents": {"kind": "markdown", "value": "**bold**"}}
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method == "workspace/symbol":
        query = msg["params"]["query"]
        uri = msg.get("_uri", "file:///unused")
        if query == "nomatch":
            result = []
        else:
            result = [
                {
                    "name": "fifo",
                    "kind": 2,
                    "location": {"uri": uri, "range": pos_range(0, 0)},
                }
            ]
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)
"""

# A server that behaves the way vhdl_ls actually does for references:
# it advertises referencesProvider, then ignores the request's
# `includeDeclaration: false` and returns the declaration anyway. The
# number of non-declaration hits is the request's `character`, so one
# fake covers both the declaration-filtering and the output-cap tests.
# textDocument/definition always answers with the declaration (line 0),
# which is how the declaration is identified for filtering.
FAKE_STUBBORN_REF_LSP = r"""#!/usr/bin/env python3
import json
import sys

if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
    print("fake-stubborn-lsp 1.0.0")
    sys.exit(0)


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get(b"content-length", b"0"))
    return json.loads(sys.stdin.buffer.read(length))


def send(obj):
    body = json.dumps(obj).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    sys.stdout.buffer.write(frame + body)
    sys.stdout.buffer.flush()


def pos_range(line, char):
    return {
        "start": {"line": line, "character": char},
        "end": {"line": line, "character": char + 3},
    }


read_message()  # initialize
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "capabilities": {
                "documentSymbolProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
            }
        },
    }
)
msg = read_message()  # initialized
assert msg is not None and msg.get("method") == "initialized", msg
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": []},
            }
        )
    elif method == "textDocument/documentSymbol":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "textDocument/definition":
        uri = msg["params"]["textDocument"]["uri"]
        # The declaration, on a different column than the reference
        # list reports it (a real server does this too).
        send(
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {"uri": uri, "range": pos_range(0, 7)},
            }
        )
    elif method == "textDocument/references":
        uri = msg["params"]["textDocument"]["uri"]
        count = max(1, msg["params"]["position"]["character"])
        # includeDeclaration is deliberately ignored.
        result = [{"uri": uri, "range": pos_range(0, 0)}]
        result += [
            {"uri": uri, "range": pos_range(i, 0)} for i in range(1, count + 1)
        ]
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)
"""

FIFO_VHDL = "entity fifo is end entity fifo;\n"
OTHER_VHDL = "entity other is end entity other;\n"
TOP_V = "module top;\nendmodule\n"


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class FakeDense:
    embedding_size = 4

    def passage_embed(self, texts, batch_size=32):
        for i, text in enumerate(texts):
            yield np.array([float(len(text)), float(i), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, query, batch_size=32):
        yield np.array([float(len(query)), 0.0, 0.0, 0.0], dtype=np.float32)


def fake_providers(config: AppConfig) -> EmbeddingProviders:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    # The reranker must be faked too, not just the dense model: otherwise
    # these tests depend on whether a real cross-encoder happens to be
    # loadable on the machine (bundled package assets, or a populated
    # embed-cache), which silently changes every score from the store's
    # RRF ranking to the model's. Making it unavailable pins the
    # documented degradation path (see RetrievalService._rerank); tests
    # that exercise reranking install their own scorer over this.
    providers.rerank = _unavailable_reranker  # type: ignore[method-assign]
    return providers


def _unavailable_reranker(query: str, texts: list[str]) -> list[float]:
    raise RuntimeError("no reranker model in the unit tests")


class FakeApp:
    """Just enough of ``VhdlRagApp``'s surface for :mod:`navigation`."""

    def __init__(
        self, config: AppConfig, git_manager: GitManager, retrieval: RetrievalService
    ) -> None:
        self.config = config
        self.git = git_manager
        self.retrieval = retrieval
        self._analyzer_statuses = None

    def analyzer_statuses(self):
        if self._analyzer_statuses is None:
            self._analyzer_statuses = build_analyzer_statuses(
                self.config.vhdl_ls_path, self.config.veridian_path
            )
        return self._analyzer_statuses


@pytest.fixture
def fake_lsp(tmp_path: Path) -> Path:
    return executable_lsp_script(tmp_path, "fake_nav_lsp", FAKE_NAV_LSP)


@pytest.fixture
def hdl_remote(tmp_path: Path) -> Path:
    up = tmp_path / "hdl-upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.vhd").write_text(FIFO_VHDL)
    (up / "rtl" / "other.vhd").write_text(OTHER_VHDL)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")
    return up


@pytest.fixture
def sv_remote(tmp_path: Path) -> Path:
    up = tmp_path / "sv-upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "top.v").write_text(TOP_V)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")
    return up


@pytest.fixture
def config(
    tmp_path: Path, hdl_remote: Path, sv_remote: Path, fake_lsp: Path
) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        vhdl_ls_path=str(fake_lsp),
        veridian_path=str(fake_lsp),
        repositories=[
            RepositoryConfig(name="hdl", url=str(hdl_remote), ref="main"),
            RepositoryConfig(name="sv", url=str(sv_remote), ref="main"),
        ],
    )


@pytest.fixture
def app(config: AppConfig):
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    git_manager = GitManager(config.repos_dir)
    states = StateStore(config.sqlite_index_path)
    providers = fake_providers(config)
    pipeline = IndexPipeline(config, git_manager, store, providers, states)
    retrieval = RetrievalService(config, git_manager, store, providers, states)
    yield FakeApp(config, git_manager, retrieval), pipeline, config
    store.close()


async def _sync_all(pipeline: IndexPipeline, config: AppConfig) -> None:
    for cfg in config.repositories:
        await pipeline.sync_repository(cfg)


# -- position validation and empty-result diagnosis (pure functions) ---------

_CONV_CORE = [
    "  bias_requant_inst : entity cnn_accel.cnn_accel_bias_requant",
    "",
    "  fifo_inst : entity work.asynchronous_fifo",
]


def test_validate_position_accepts_every_real_position() -> None:
    _validate_position("a.vhd", _CONV_CORE, 0, 0)
    _validate_position("a.vhd", _CONV_CORE, 0, len(_CONV_CORE[0]))  # end of line
    _validate_position("a.vhd", _CONV_CORE, 1, 0)  # a blank line
    _validate_position("a.vhd", _CONV_CORE, 2, 5)


def test_validate_position_names_the_files_real_extent() -> None:
    with pytest.raises(RetrievalError) as excinfo:
        _validate_position("a.vhd", _CONV_CORE, 99999, 0)
    message = str(excinfo.value)
    assert "past the end of 'a.vhd'" in message
    assert "3 lines" in message
    assert "last addressable line is 2" in message


def test_validate_position_rejects_negative_and_overlong_positions() -> None:
    with pytest.raises(RetrievalError, match="must not be negative"):
        _validate_position("a.vhd", _CONV_CORE, -1, 0)
    with pytest.raises(RetrievalError, match="must not be negative"):
        _validate_position("a.vhd", _CONV_CORE, 0, -1)
    with pytest.raises(RetrievalError) as excinfo:
        _validate_position("a.vhd", _CONV_CORE, 0, 500)
    assert "past the end of line 0" in str(excinfo.value)
    assert f"{len(_CONV_CORE[0])} characters long" in str(excinfo.value)
    with pytest.raises(RetrievalError, match="is empty"):
        _validate_position("a.vhd", [], 0, 0)


def test_selected_name_at_picks_up_the_whole_dotted_name() -> None:
    # Anywhere inside 'cnn_accel.cnn_accel_bias_requant' yields all of it.
    assert _selected_name_at(_CONV_CORE, 0, 30) == "cnn_accel.cnn_accel_bias_requant"
    assert _selected_name_at(_CONV_CORE, 0, 45) == "cnn_accel.cnn_accel_bias_requant"
    assert _selected_name_at(_CONV_CORE, 0, 2) == "bias_requant_inst"
    assert _selected_name_at(_CONV_CORE, 0, 0) == ""  # leading whitespace
    assert _selected_name_at(_CONV_CORE, 1, 0) == ""  # blank line
    assert _selected_name_at(_CONV_CORE, 99, 0) == ""


def test_no_result_message_blames_the_library_for_a_qualified_name() -> None:
    """The dominant real failure: a library-qualified instantiation that
    no configured library can resolve. Saying only "No definition found"
    makes a caller conclude the entity does not exist."""
    out = _no_result_message("definition", _CONV_CORE, 0, 30)
    assert out.startswith("No definition found at that position.")
    assert "'cnn_accel.cnn_accel_bias_requant'" in out
    assert "library-qualified" in out
    assert "'cnn_accel'" in out
    assert "vhdl_ls.toml" in out
    assert "not a 'not declared' one" in out


def test_no_result_message_treats_work_as_the_files_own_library() -> None:
    out = _no_result_message("definition", _CONV_CORE, 2, 25)
    assert "work.asynchronous_fifo" in out
    # 'work' always resolves, so pointing at library config would mislead.
    assert "library-qualified" not in out
    assert "vhdl_ls.toml" not in out


def test_no_result_message_calls_out_a_blank_line() -> None:
    out = _no_result_message("hover information", _CONV_CORE, 1, 0)
    assert out.startswith("No hover information found at that position.")
    assert "Line 1 is blank" in out
    assert "line 2 in an editor" in out


def test_no_result_message_calls_out_a_position_off_any_identifier() -> None:
    out = _no_result_message("references", _CONV_CORE, 0, 0)
    assert "is not part of an identifier" in out


def test_no_result_message_for_a_plain_unresolved_identifier() -> None:
    out = _no_result_message("definition", _CONV_CORE, 0, 2)
    assert "'bias_requant_inst'" in out
    assert "vhdl_ls configuration does not declare" in out


# -- error paths that never need a session -----------------------------------


async def test_find_definition_rejects_non_hdl_file(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="not a VHDL/Verilog/SystemVerilog"):
        await find_definition(fake_app, "hdl", "README.md", 0, 0)


async def test_find_definition_rejects_unknown_repository(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="unknown repository"):
        await find_definition(fake_app, "no-such-repo", "rtl/fifo.vhd", 0, 0)


async def test_find_definition_rejects_missing_file(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="not found"):
        await find_definition(fake_app, "hdl", "rtl/does-not-exist.vhd", 0, 0)


async def test_find_definition_reports_unavailable_analyzer(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    fake_app.config = config.model_copy(update={"vhdl_ls_path": "no-such-binary-xyz"})
    fake_app._analyzer_statuses = None
    with pytest.raises(RetrievalError, match="not available"):
        await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 0, 0)


# -- end-to-end navigation over the fake server ------------------------------


async def test_find_definition_single_location_same_file(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 0, 1)
    assert "hdl:rtl/fifo.vhd:3:1" in out  # 0-based line 2 -> 1-based line 3
    assert "entity fifo" in out


async def test_find_definition_cross_file(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 0, 2)
    assert "hdl:rtl/other.vhd:1:1" in out
    assert "entity other" in out


async def test_find_definition_location_link_uses_selection_range(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 0, 3)
    assert "hdl:rtl/fifo.vhd:5:3" in out  # targetSelectionRange (4, 2) -> 1-based


async def test_find_definition_no_result(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 0, 0)
    assert out.startswith("No definition found at that position.")
    # Not just the flat sentence: it names the symbol it looked at, so
    # the caller can tell "not declared" from "wrong position".
    assert "'entity'" in out


async def test_find_definition_rejects_a_position_past_the_end_of_the_file(
    app,
) -> None:
    """A position past EOF used to come back as the same flat "No
    definition found at that position." as a genuinely undefined
    symbol, so a caller concluded the symbol did not exist."""
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="past the end of"):
        await find_definition(fake_app, "hdl", "rtl/fifo.vhd", 99999, 0)


async def test_find_references_rejects_a_negative_position(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="must not be negative"):
        await find_references(fake_app, "hdl", "rtl/fifo.vhd", -1, 0)


async def test_hover_info_rejects_a_character_past_the_end_of_the_line(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="past the end of line 0"):
        await hover_info(fake_app, "hdl", "rtl/fifo.vhd", 0, 500)


async def test_find_references_include_and_exclude_declaration(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with_decl = await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 0)
    assert with_decl.count("hdl:rtl/fifo.vhd:") == 2
    without_decl = await find_references(
        fake_app, "hdl", "rtl/fifo.vhd", 0, 0, include_declaration=False
    )
    assert without_decl.count("hdl:rtl/fifo.vhd:") == 1


@pytest.fixture
def stubborn_app(app, tmp_path: Path):
    """The same app, but talking to a server that ignores
    ``includeDeclaration`` (as vhdl_ls does) and can return an arbitrary
    number of references."""
    fake_app, pipeline, config = app
    binary = executable_lsp_script(tmp_path, "fake_stubborn_lsp", FAKE_STUBBORN_REF_LSP)
    fake_app.config = config.model_copy(update={"vhdl_ls_path": str(binary)})
    fake_app._analyzer_statuses = None
    return fake_app, pipeline, config


async def test_find_references_filters_declaration_the_server_keeps(
    stubborn_app,
) -> None:
    """``include_declaration=False`` used to be a no-op: vhdl_ls ignores
    the LSP flag and returns the declaration regardless, so the two
    responses were byte-identical. The declaration is now filtered out
    here instead."""
    fake_app, pipeline, config = stubborn_app
    await _sync_all(pipeline, config)
    with_decl = await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 1)
    assert with_decl.count("hdl:rtl/fifo.vhd:") == 2
    assert "hdl:rtl/fifo.vhd:1:1" in with_decl  # the declaration (LSP line 0)

    without_decl = await find_references(
        fake_app, "hdl", "rtl/fifo.vhd", 0, 1, include_declaration=False
    )
    assert without_decl != with_decl
    assert without_decl.count("hdl:rtl/fifo.vhd:") == 1
    assert "hdl:rtl/fifo.vhd:1:1" not in without_decl
    assert "hdl:rtl/fifo.vhd:2:1" in without_decl


async def test_find_references_caps_output_and_says_so(stubborn_app) -> None:
    """A common signal name has unboundedly many references, each
    rendered with ~5 lines of context, so the output is capped and the
    true total reported."""
    fake_app, pipeline, config = stubborn_app
    await _sync_all(pipeline, config)
    # 30 references + the declaration.
    out = await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 30, limit=5)
    assert out.count("hdl:rtl/fifo.vhd:") == 5
    assert "31 references found; showing the first 5" in out

    # The default cap applies without an explicit limit.
    out = await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 30)
    assert out.count("hdl:rtl/fifo.vhd:") == REFERENCE_LIMIT
    assert f"showing the first {REFERENCE_LIMIT}" in out

    # Under the cap: no note at all.
    out = await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 1, limit=5)
    assert "references found; showing" not in out


async def test_find_references_rejects_a_zero_limit(stubborn_app) -> None:
    fake_app, pipeline, config = stubborn_app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="limit must be at least 1"):
        await find_references(fake_app, "hdl", "rtl/fifo.vhd", 0, 1, limit=0)


async def test_hover_info_plain_and_markup_and_none(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    assert await hover_info(fake_app, "hdl", "rtl/fifo.vhd", 0, 1) == "plain hover text"
    assert await hover_info(fake_app, "hdl", "rtl/fifo.vhd", 0, 2) == "**bold**"
    out = await hover_info(fake_app, "hdl", "rtl/fifo.vhd", 0, 0)
    assert out.startswith("No hover information found at that position.")


async def test_find_symbol_within_one_repository(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_symbol(fake_app, "hdl", "fifo")
    assert "module fifo" in out
    assert "hdl:" in out


async def test_find_symbol_no_match(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_symbol(fake_app, "hdl", "nomatch")
    assert "No symbols matching" in out
    assert "search_hdl" in out


async def test_find_symbol_across_all_repositories_when_unset(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    out = await find_symbol(fake_app, None, "fifo", limit=8)
    # The VHDL repo's vhdl_ls session and the Verilog repo's Veridian
    # session both answer, so both repositories should appear.
    assert "hdl:" in out
    assert "sv:" in out


async def test_find_symbol_rejects_unknown_repository(app) -> None:
    """Every other tool errors on an unknown repository name. find_symbol
    used to swallow it in its per-repository error containment and answer
    "No symbols matching ...", which reads as "the symbol does not
    exist" rather than "you named a repository that is not
    configured"."""
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="unknown repository"):
        await find_symbol(fake_app, "no-such-repo", "fifo")


async def test_find_symbol_empty_query_rejected(app) -> None:
    fake_app, pipeline, config = app
    await _sync_all(pipeline, config)
    with pytest.raises(RetrievalError, match="must not be empty"):
        await find_symbol(fake_app, "hdl", "   ")


def test_uri_to_path_windows_drive_letter() -> None:
    # ``file:///C:/repo/rtl/fifo.vhd`` is what Path.as_uri() (and every
    # LSP server) emits for a Windows path — the POSIX-style leading
    # slash in front of the drive letter must be stripped, or
    # relative_to() against a real Windows repo_dir never matches (the
    # bug this regression test covers: it doesn't depend on running on
    # Windows, since the URI shape itself is platform-independent).
    assert _uri_to_path("file:///C:/repo/rtl/fifo.vhd") == Path("C:/repo/rtl/fifo.vhd")
    assert _uri_to_path("file:///c:/a%20b/x.vhd") == Path("c:/a b/x.vhd")


def test_uri_to_path_posix_unchanged() -> None:
    assert _uri_to_path("file:///home/user/repo/fifo.vhd") == Path(
        "/home/user/repo/fifo.vhd"
    )
