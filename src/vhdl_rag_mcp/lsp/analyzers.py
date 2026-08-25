"""HDL analyzer discovery, status, and client factory.

The two HDL language servers are **optional external dependencies**:
vhdl_ls (VHDL) and Veridian (Verilog/SystemVerilog). Neither is
bundled, vendored, or installed by this application. Each is located
by an explicitly configured path first, then on ``PATH``; the resolved
binary is probed for a self-reported version. The per-analyzer
:class:`AnalyzerStatus` (available/path/version/mode/error) is what
the server reports and the indexing pipeline consults: when an
analyzer is unavailable its files fall back to structural parsing, so
a missing analyzer degrades gracefully instead of failing.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import RepositoryConfig
from .client import LspClient, VhdlLsp, default_libraries_dir, server_version
from .veridian import VeridianLsp

logger = logging.getLogger(__name__)

MODE_LSP = "lsp"
MODE_FALLBACK = "fallback"


@dataclass(frozen=True)
class AnalyzerStatus:
    """Availability of one optional HDL analyzer.

    ``mode`` is :data:`MODE_LSP` when the binary resolved (files are
    chunked from its documentSymbol tree) and :data:`MODE_FALLBACK`
    otherwise (structural/generic parsing). ``path``/``version`` are
    None when unavailable; ``error`` carries the reason.
    """

    name: str
    available: bool
    path: str | None
    version: str | None
    mode: str
    error: str | None

    @classmethod
    def unavailable(cls, name: str, error: str) -> AnalyzerStatus:
        return cls(name, False, None, None, MODE_FALLBACK, error)


def resolve_binary(
    configured: str | None, fallback_name: str
) -> tuple[str | None, str | None]:
    """Locate a language-server binary: (resolved path, error).

    The explicitly configured value (a PATH name or an absolute/relative
    path) wins; when unset or unresolvable, the default name on ``PATH``
    is tried. Exactly one of the two return values is None.
    """
    configured = (configured or "").strip()
    if configured:
        found = shutil.which(configured)
        if found is not None:
            return found, None
        return None, f"configured path {configured!r} was not found"
    found = shutil.which(fallback_name)
    if found is not None:
        return found, None
    return None, (
        f"{fallback_name} not found on PATH; set the corresponding path "
        "config option or install it"
    )


def analyzer_status(name: str, configured: str | None) -> AnalyzerStatus:
    """Probe one analyzer: discovery + version, never raises."""
    path, error = resolve_binary(configured, name)
    if path is None:
        return AnalyzerStatus.unavailable(name, error or f"{name} not found")
    version = server_version(path)
    return AnalyzerStatus(
        name=name,
        available=True,
        path=path,
        version=version,
        mode=MODE_LSP,
        error=None,
    )


def build_analyzer_statuses(
    vhdl_ls_path: str | None, veridian_path: str | None
) -> dict[str, AnalyzerStatus]:
    """Status for every HDL analyzer (keys: analyzer names)."""
    return {
        "vhdl_ls": analyzer_status("vhdl_ls", vhdl_ls_path),
        "veridian": analyzer_status("veridian", veridian_path),
    }


def build_client(
    status: AnalyzerStatus,
    repo_cfg: RepositoryConfig,
    workspace: Path,
) -> LspClient | None:
    """The LSP client for one analyzer, or None when it is unavailable.

    Wires the repository's per-analyzer config hooks (``vhdl_ls_hook``
    / ``veridian_hook``) and, for vhdl_ls, the ``vhdl_libraries``
    directory shipped next to the binary.
    """
    if not status.available or not status.path:
        return None
    if status.name == "vhdl_ls":
        return VhdlLsp(
            status.path,
            workspace,
            libraries_dir=default_libraries_dir(status.path),
            vhdl_ls_hook=repo_cfg.vhdl_ls_hook,
        )
    if status.name == "veridian":
        return VeridianLsp(status.path, workspace, config_hook=repo_cfg.veridian_hook)
    raise ValueError(f"unknown analyzer: {status.name!r}")
