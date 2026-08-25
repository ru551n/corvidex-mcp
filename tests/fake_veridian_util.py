"""Shared helper: a fake Veridian LSP server for pipeline-level tests.

Speaks the same Content-Length framing as the real Veridian and mirrors
its observed behavior: plain-name documentSymbol trees with standard
LSP kinds (modules/packages/interfaces as top-level siblings, functions
and tasks as children — always blocks are NOT in the tree), and
slang-style severity-based diagnostics without codes. Symbol trees and
diagnostics are provided per file, keyed by basename.
"""

from __future__ import annotations

import json
from pathlib import Path

from fake_lsp_util import executable_lsp_script

_TEMPLATE = r"""#!/usr/bin/env python3
import json
import sys


if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
    print("veridian 9.9.9-test")
    sys.exit(0)

SYMBOLS = __SYMBOLS__
DIAGNOSTICS = __DIAGNOSTICS__


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


def base(uri):
    return uri.rsplit("/", 1)[-1]


def handle(method, msg):
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": DIAGNOSTICS.get(base(uri), [])},
            }
        )
    elif method == "textDocument/documentSymbol":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": SYMBOLS.get(base(uri)),
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
    handle(msg.get("method"), msg)
"""


def fake_veridian(
    tmp_path: Path,
    name: str,
    symbols_by_file: dict[str, list[dict]],
    diagnostics_by_file: dict[str, list[dict]] | None = None,
) -> Path:
    """Write a fake Veridian binary answering per-file symbol trees."""
    source = _TEMPLATE.replace("__SYMBOLS__", json.dumps(symbols_by_file)).replace(
        "__DIAGNOSTICS__", json.dumps(diagnostics_by_file or {})
    )
    return executable_lsp_script(tmp_path, name, source)
