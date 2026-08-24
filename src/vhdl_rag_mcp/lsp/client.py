"""vhdl_ls language-server client (LSP over stdio, Content-Length framing).

One :class:`VhdlLsp` instance per repository, for one sync run:

- the workspace is the repository working tree (checked out at the target
  commit by :mod:`vhdl_rag_mcp.git_manager`);
- a ``vhdl_ls.toml`` is generated in the workspace root unless the
  repository already provides its own (in which case it is respected and
  left in place);
- after opening the changed files the client waits for the server to go
  quiet (vhdl_ls pushes ``publishDiagnostics`` for every workspace file it
  analyzes, but sends nothing for clean files, so a per-file wait is not
  possible);
- ``documentSymbol`` results are parsed into a plain
  :class:`SymbolInfo` tree (hierarchical, as advertised during
  ``initialize``).

All failures are contained: a hung or broken language server surfaces as
:class:`LspError` and the chunker falls back to structural parsing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: How long to wait for the server to stop emitting diagnostics.
QUIET_TIMEOUT = 20.0
#: Diagnostics silence that counts as "the analysis is done".
QUIET_WINDOW = 1.5
#: Per-request timeout for ordinary LSP requests.
REQUEST_TIMEOUT = 30.0
#: Timeout for the initialize handshake.
INITIALIZE_TIMEOUT = 60.0

DEFAULTLIB_GLOBS = ["**/*.vhd", "**/*.vhdl"]


class LspError(RuntimeError):
    """The language server failed a request or closed the connection."""


class LspTimeout(LspError):
    """A language-server request did not answer in time."""


@dataclass(frozen=True)
class SymbolInfo:
    """One LSP document symbol with its (0-based, inclusive) line range."""

    name: str
    kind: int
    start_line: int
    end_line: int
    children: tuple[SymbolInfo, ...] = ()


@dataclass(frozen=True)
class DiagnosticInfo:
    """One LSP diagnostic (code such as ``syntax_error``, ``unresolved``)."""

    code: str
    message: str
    severity: int
    start_line: int
    end_line: int


def default_libraries_dir(binary: str) -> Path | None:
    """Locate the ``vhdl_libraries`` directory shipped next to the binary.

    The official distribution layout is ``<root>/bin/vhdl_ls`` plus
    ``<root>/vhdl_libraries``. Returns None when it cannot be found (the
    server then runs without ``-l``).
    """
    bin_path = Path(binary).expanduser()
    for candidate in (
        bin_path.parent / "vhdl_libraries",
        bin_path.parent.parent / "vhdl_libraries",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _parse_symbols(items: Any) -> tuple[SymbolInfo, ...]:
    if not isinstance(items, list):
        return ()
    out: list[SymbolInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rng = item.get("range")
        if not isinstance(rng, dict):
            # Some servers report "location" instead of "range".
            loc = item.get("location")
            if isinstance(loc, dict) and isinstance(loc.get("range"), dict):
                rng = loc["range"]
        if not isinstance(rng, dict):
            continue
        try:
            out.append(
                SymbolInfo(
                    name=str(item.get("name", "")),
                    kind=int(item.get("kind", 0)),
                    start_line=int(rng["start"]["line"]),
                    end_line=int(rng["end"]["line"]),
                    children=_parse_symbols(item.get("children")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


class VhdlLsp:
    """Async client for one vhdl_ls process rooted at one workspace."""

    def __init__(
        self, binary: str, workspace: Path, libraries_dir: Path | None = None
    ) -> None:
        self._binary = binary
        self._workspace = workspace
        self._libraries = libraries_dir
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._diagnostics: dict[str, list[DiagnosticInfo]] = {}
        self._quiet_event = asyncio.Event()
        self._supports_document_symbol = False
        self._owns_workspace_config = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Spawn vhdl_ls, perform the LSP handshake, and open the workspace."""
        self._ensure_workspace_config()
        args = [self._binary]
        if self._libraries is not None and self._libraries.is_dir():
            args += ["-l", str(self._libraries)]
        args.append("--silent")
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self._workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader = asyncio.create_task(self._read_loop())
        try:
            result = await self._request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": self._workspace.as_uri(),
                    "capabilities": {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True
                            }
                        }
                    },
                    "workspaceFolders": [
                        {"uri": self._workspace.as_uri(), "name": self._workspace.name}
                    ],
                },
                timeout=INITIALIZE_TIMEOUT,
            )
            caps = result.get("capabilities") if isinstance(result, dict) else None
            self._supports_document_symbol = bool(
                (caps or {}).get("documentSymbolProvider")
            )
            await self._notify("initialized", {})
            logger.info("vhdl_ls started for %s", self._workspace)
        except BaseException:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown; never raises."""
        if self._proc is not None:
            with contextlib.suppress(Exception):
                await self._request("shutdown", None, timeout=10.0)
                await self._notify("exit", {})
            # Give the server a moment to exit cleanly after "exit".
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proc.wait(), 5.0)
            if self._proc.returncode is None:
                self._proc.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._proc.wait(), 5.0)
            if self._proc.returncode is None:
                self._proc.kill()
                with contextlib.suppress(Exception):
                    await self._proc.wait()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._reader
            self._reader = None
        self._fail_pending(LspError("language server shut down"))
        if self._owns_workspace_config:
            self._workspace.joinpath("vhdl_ls.toml").unlink(missing_ok=True)

    def _ensure_workspace_config(self) -> None:
        config_path = self._workspace / "vhdl_ls.toml"
        if config_path.exists():
            logger.info("using repository-provided vhdl_ls.toml in %s", self._workspace)
            return
        self._owns_workspace_config = True
        files_list = ", ".join(f"'{glob}'" for glob in DEFAULTLIB_GLOBS)
        lines = ["[libraries.defaultlib]", f"files = [{files_list}]", ""]
        if self._libraries is not None and self._libraries.is_dir():
            lib = str(self._libraries)
            ieee_files = ", ".join(
                f"'{lib}/{name}/*.vhdl'"
                for name in ("ieee2008", "synopsys", "vital2000")
            )
            lines += [
                "[libraries.std]",
                f"files = ['{lib}/std/*.vhd']",
                "is_third_party = true",
                "",
                "[libraries.ieee]",
                f"files = [{ieee_files}]",
                "is_third_party = true",
            ]
        config_path.write_text("\n".join(lines), encoding="utf-8")

    # -- document flow --------------------------------------------------------

    async def open_document(self, path: Path) -> None:
        """didOpen one file (content is read from the working tree)."""
        uri = path.as_uri()
        text = path.read_text(encoding="utf-8", errors="replace")
        await self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "vhdl",
                    "version": 1,
                    "text": text,
                }
            },
        )

    async def close_document(self, path: Path) -> None:
        await self._notify(
            "textDocument/didClose", {"textDocument": {"uri": path.as_uri()}}
        )

    async def wait_until_quiet(self, timeout: float = QUIET_TIMEOUT) -> None:
        """Wait until the server stops emitting diagnostics.

        vhdl_ls pushes ``publishDiagnostics`` for every file it analyzes
        (including files we did not open) and emits nothing for clean
        files, so the analysis is "done" when no new diagnostics have
        arrived for ``QUIET_WINDOW`` seconds (bounded by ``timeout``).
        """
        self._quiet_event.clear()
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "vhdl_ls diagnostics not quiet after %.0fs in %s",
                    timeout,
                    self._workspace,
                )
                return
            self._quiet_event.clear()
            try:
                await asyncio.wait_for(
                    self._quiet_event.wait(), min(QUIET_WINDOW, remaining)
                )
                # New diagnostics arrived: wait for the silence again.
            except TimeoutError:
                return

    def diagnostics_for(self, path: Path) -> tuple[DiagnosticInfo, ...]:
        """Diagnostics collected so far for one file (empty when clean)."""
        return tuple(self._diagnostics.get(path.as_uri(), ()))

    def has_syntax_error(self, path: Path) -> bool:
        return any(d.code == "syntax_error" for d in self.diagnostics_for(path))

    @property
    def supports_document_symbol(self) -> bool:
        return self._supports_document_symbol

    async def document_symbols(self, path: Path) -> tuple[SymbolInfo, ...]:
        """Hierarchical document symbols for one file; () when unavailable."""
        if not self._supports_document_symbol:
            return ()
        try:
            result = await self._request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": path.as_uri()}},
            )
        except LspError as exc:
            logger.warning("documentSymbol failed for %s: %s", path, exc)
            return ()
        return _parse_symbols(result)

    # -- transport --------------------------------------------------------------

    async def _request(
        self, method: str, params: Any, timeout: float = REQUEST_TIMEOUT
    ) -> Any:
        assert self._proc is not None and self._proc.stdin is not None
        async with self._request_lock:
            self._next_id += 1
            request_id = self._next_id
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            try:
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
            except LspError as exc:
                self._pending.pop(request_id, None)
                future.cancel()
                raise exc
            try:
                return await asyncio.wait_for(future, timeout)
            except TimeoutError:
                self._pending.pop(request_id, None)
                future.cancel()
                raise LspTimeout(f"{method} timed out after {timeout:.0f}s") from None

    async def _notify(self, method: str, params: Any) -> None:
        assert self._proc is not None
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(obj).encode("utf-8")
        frame = (
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        try:
            async with self._write_lock:
                self._proc.stdin.write(frame)
                await self._proc.stdin.drain()
        except Exception as exc:
            raise LspError(f"failed to send to language server: {exc}") from exc

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                try:
                    header = await stream.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    break
                length = _parse_content_length(header)
                if length is None:
                    continue
                try:
                    body = await stream.readexactly(length)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    break
                try:
                    message = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning("ignoring malformed LSP message from vhdl_ls")
                    continue
                self._dispatch(message)
        finally:
            self._fail_pending(LspError("language server connection closed"))

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "id" in message and ("result" in message or "error" in message):
            request_id = message["id"]
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"]
                detail = error.get("message") if isinstance(error, dict) else str(error)
                future.set_exception(LspError(f"LSP error: {detail}"))
            else:
                future.set_result(message["result"])
        elif "method" in message:
            self._handle_notification(message["method"], message.get("params"))

    def _handle_notification(self, method: str, params: Any) -> None:
        if method != "textDocument/publishDiagnostics" or not isinstance(params, dict):
            return
        uri = str(params.get("uri", ""))
        diagnostics: list[DiagnosticInfo] = []
        for item in params.get("diagnostics") or []:
            if not isinstance(item, dict):
                continue
            rng = item.get("range")
            if not isinstance(rng, dict):
                continue
            try:
                diagnostics.append(
                    DiagnosticInfo(
                        code=str(item.get("code") or ""),
                        message=str(item.get("message") or ""),
                        severity=int(item.get("severity", 1)),
                        start_line=int(rng["start"]["line"]),
                        end_line=int(rng["end"]["line"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._diagnostics[uri] = diagnostics
        self._quiet_event.set()

    def _fail_pending(self, error: LspError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()


def _parse_content_length(header: bytes) -> int | None:
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            with contextlib.suppress(ValueError):
                return int(line.split(b":", 1)[1].strip())
    return None


# Kept for import convenience (used by tests).
__all__ = [
    "DiagnosticInfo",
    "LspError",
    "LspTimeout",
    "SymbolInfo",
    "VhdlLsp",
    "default_libraries_dir",
]
