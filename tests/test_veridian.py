"""Tests for the Veridian analyzer: client, discovery, status, factory.

Runs fully offline against fake Veridian servers (small Python scripts
speaking the same Content-Length framing, mirroring Veridian's
observed behavior: plain-name documentSymbol trees, slang-style
severity-based diagnostics without codes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fake_lsp_util import executable_lsp_script

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.lsp import (
    MODE_FALLBACK,
    MODE_LSP,
    AnalyzerStatus,
    LspClient,
    VeridianLsp,
    VhdlLsp,
    analyzer_status,
    build_analyzer_statuses,
    build_client,
    resolve_binary,
    server_version,
)

FAKE_VERIDIAN = r"""#!/usr/bin/env python3
import json
import os
import sys


if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
    print("veridian 9.9.9-test")
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


def symbol_range(start_line, end_line):
    return {
        "start": {"line": start_line, "character": 0},
        "end": {"line": end_line, "character": 3},
    }


def symbols():
    # Veridian style: PLAIN names, standard LSP kinds (no kind prefix).
    return [
        {
            "name": "fifo",
            "kind": 2,
            "range": symbol_range(0, 9),
            "children": [
                {
                    "name": "DEPTH",
                    "kind": 26,
                    "range": symbol_range(1, 1),
                    "children": [],
                },
                {
                    "name": "clk",
                    "kind": 7,
                    "range": symbol_range(2, 2),
                    "children": [],
                },
                {
                    "name": "r",
                    "kind": 13,
                    "range": symbol_range(3, 3),
                    "children": [],
                },
                {
                    "name": "add8",
                    "kind": 12,
                    "range": symbol_range(4, 8),
                    "children": [],
                },
            ],
        }
    ]


def handle(method, msg):
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        record = os.environ.get("FAKE_RECORD")
        if record:
            with open(record, "a") as fh:
                fh.write(msg["params"]["textDocument"]["languageId"] + "\n")
        diags = (
            [
                {
                    "source": "slang",
                    "message": " expected 'endmodule'",
                    "severity": 1,
                    "range": symbol_range(0, 0),
                }
            ]
            if "badfile" in uri
            else []
        )
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": diags},
            }
        )
    elif method == "textDocument/documentSymbol":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": symbols() if "badfile" not in uri else None,
            }
        )
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)


read_message()  # initialize
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"capabilities": {"documentSymbolProvider": True}},
    }
)
msg = read_message()
assert msg is not None and msg.get("method") == "initialized", msg
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method:
        handle(method, msg)
"""

# Veridian dies right after the handshake (crash on malformed input).
FAKE_VERIDIAN_DEAD = r"""#!/usr/bin/env python3
import json
import os
import sys


if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
    print("veridian 9.9.9-test")
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


read_message()  # initialize
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"capabilities": {"documentSymbolProvider": True}},
    }
)
read_message()  # initialized
os._exit(1)  # crash like Veridian does on malformed files
"""


@pytest.fixture
def fake_veridian(tmp_path: Path) -> Path:
    return executable_lsp_script(tmp_path, "fake_veridian", FAKE_VERIDIAN)


@pytest.fixture
def dead_veridian(tmp_path: Path) -> Path:
    return executable_lsp_script(tmp_path, "dead_veridian", FAKE_VERIDIAN_DEAD)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "fifo.sv").write_text("module fifo; endmodule\n")
    (ws / "badfile.sv").write_text("module broken (\n")
    return ws


# -- client: config file lifecycle ------------------------------------------


async def test_default_config_written_and_removed(
    fake_veridian: Path, workspace: Path
) -> None:
    lsp = VeridianLsp(str(fake_veridian), workspace)
    try:
        await lsp.start()
        text = (workspace / "veridian.yaml").read_text()
        assert "auto_search_workdir: true" in text
        assert "include_dirs:" in text
        assert "source_dirs:" in text
        assert "log_level: Warn" in text
    finally:
        await lsp.shutdown()
    # Server-generated config is cleaned up on shutdown.
    assert not (workspace / "veridian.yaml").exists()


async def test_repository_config_respected_and_kept(
    fake_veridian: Path, workspace: Path
) -> None:
    (workspace / "veridian.yaml").write_text("# repo-owned\nlog_level: Error\n")
    lsp = VeridianLsp(str(fake_veridian), workspace)
    try:
        await lsp.start()
        assert (
            workspace / "veridian.yaml"
        ).read_text() == "# repo-owned\nlog_level: Error\n"
    finally:
        await lsp.shutdown()
    assert (
        workspace / "veridian.yaml"
    ).read_text() == "# repo-owned\nlog_level: Error\n"


async def test_hook_generates_config_and_is_not_removed(
    fake_veridian: Path, workspace: Path, tmp_path: Path
) -> None:
    script = tmp_path / "veridian_hook.py"
    script.write_text(
        "import pathlib\n"
        'pathlib.Path("veridian.yaml").write_text("# hook-owned" + chr(10))\n',
        encoding="utf-8",
    )
    lsp = VeridianLsp(
        str(fake_veridian), workspace, config_hook=f"{sys.executable} {script}"
    )
    try:
        await lsp.start()
        assert (workspace / "veridian.yaml").read_text() == "# hook-owned\n"
    finally:
        await lsp.shutdown()
    # Hook output is owned by the hook: never removed.
    assert (workspace / "veridian.yaml").read_text() == "# hook-owned\n"


async def test_hook_failure_falls_back_to_default(
    fake_veridian: Path, workspace: Path
) -> None:
    lsp = VeridianLsp(str(fake_veridian), workspace, config_hook="exit 1")
    try:
        await lsp.start()
        assert "auto_search_workdir: true" in (workspace / "veridian.yaml").read_text()
    finally:
        await lsp.shutdown()
    assert not (workspace / "veridian.yaml").exists()


# -- client: documents, symbols, diagnostics ---------------------------------


async def test_symbol_tree_and_diagnostics(
    fake_veridian: Path, workspace: Path
) -> None:
    lsp = VeridianLsp(str(fake_veridian), workspace)
    try:
        await lsp.start()
        await lsp.open_document(workspace / "fifo.sv")
        await lsp.open_document(workspace / "badfile.sv")
        await lsp.wait_until_quiet(timeout=5.0)
        symbols = await lsp.document_symbols(workspace / "fifo.sv")
        assert len(symbols) == 1
        module = symbols[0]
        # Plain names (no kind prefix), standard kinds.
        assert (module.name, module.kind) == ("fifo", 2)
        children = {c.name: c.kind for c in module.children}
        assert children == {"DEPTH": 26, "clk": 7, "r": 13, "add8": 12}
        # slang diagnostics: no code field; severity 1 marks the error.
        assert not lsp.has_syntax_error(workspace / "fifo.sv")
        assert lsp.has_syntax_error(workspace / "badfile.sv")
        # A file with errors yields no symbol tree.
        assert await lsp.document_symbols(workspace / "badfile.sv") == ()
    finally:
        await lsp.shutdown()


async def test_open_document_uses_verilog_language_id(
    fake_veridian: Path,
    workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = tmp_path / "langids"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    lsp = VeridianLsp(str(fake_veridian), workspace)
    try:
        await lsp.start()
        await lsp.open_document(workspace / "fifo.sv")
        await lsp.wait_until_quiet(timeout=5.0)
    finally:
        await lsp.shutdown()
    assert record.read_text().splitlines() == ["verilog"]


async def test_dead_server_degrades_to_empty_symbols(
    dead_veridian: Path, workspace: Path
) -> None:
    lsp = VeridianLsp(str(dead_veridian), workspace)
    try:
        await lsp.start()
        # The server crashes right after the handshake.
        await lsp.open_document(workspace / "fifo.sv")  # must not raise
        await lsp.wait_until_quiet(timeout=2.0)  # must not wait 20s+
        assert not lsp.server_alive
        assert await lsp.document_symbols(workspace / "fifo.sv") == ()
        assert not lsp.has_syntax_error(workspace / "fifo.sv")
    finally:
        await lsp.shutdown()


# -- discovery / status / factory --------------------------------------------


def test_analyzer_status_available(fake_veridian: Path) -> None:
    status = analyzer_status("veridian", str(fake_veridian))
    assert status.available
    assert status.path == str(fake_veridian)
    assert status.version == "veridian 9.9.9-test"
    assert status.mode == MODE_LSP
    assert status.error is None


def test_analyzer_status_unavailable() -> None:
    status = analyzer_status("veridian", "/nonexistent/veridian")
    assert not status.available
    assert status.mode == MODE_FALLBACK
    assert status.path is None
    assert status.version is None
    assert "was not found" in (status.error or "")


def test_resolve_binary_prefers_configured() -> None:
    path, error = resolve_binary(sys.executable, "no-such-binary-xyz")
    assert path == sys.executable and error is None
    path, error = resolve_binary(None, "no-such-binary-xyz")
    assert path is None and "no-such-binary-xyz" in (error or "")
    path, error = resolve_binary("   ", "no-such-binary-xyz")
    assert path is None


def test_build_analyzer_statuses_mixed(
    fake_veridian: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make the PATH lookup deterministic: only the fake binary resolves.
    def fake_which(name: str) -> str | None:
        return str(fake_veridian) if name == str(fake_veridian) else None

    monkeypatch.setattr("corvidex_mcp.lsp.analyzers.shutil.which", fake_which)
    statuses = build_analyzer_statuses(str(fake_veridian), "/no/such/veridian")
    assert statuses["vhdl_ls"].mode == MODE_LSP
    assert statuses["vhdl_ls"].path == str(fake_veridian)
    assert statuses["veridian"].mode == MODE_FALLBACK
    assert statuses["veridian"].error is not None


def test_build_client_per_analyzer(fake_veridian: Path) -> None:
    repo_cfg = RepositoryConfig(name="r", url="u", veridian_hook="make ver")
    veridian_status = analyzer_status("veridian", str(fake_veridian))
    client = build_client(veridian_status, repo_cfg, Path("/ws"))
    assert isinstance(client, VeridianLsp)

    vhdl_status = analyzer_status("vhdl_ls", str(fake_veridian))
    assert isinstance(build_client(vhdl_status, repo_cfg, Path("/ws")), VhdlLsp)

    dead_status = AnalyzerStatus.unavailable("veridian", "gone")
    assert build_client(dead_status, repo_cfg, Path("/ws")) is None


def test_server_version_unrunnable() -> None:
    assert server_version("/nonexistent/binary") is None


def test_client_base_rejects_no_config_server() -> None:
    class Bare(LspClient):
        pass

    client = Bare("veridian", Path("/ws"))
    assert client.config_name is None
    assert client.default_config_text() is None
