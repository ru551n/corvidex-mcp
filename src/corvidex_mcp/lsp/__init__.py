"""LSP clients over stdio (see :mod:`corvidex_mcp.lsp.client`)."""

from .analyzers import (
    MODE_FALLBACK,
    MODE_LSP,
    AnalyzerStatus,
    analyzer_status,
    build_analyzer_statuses,
    resolve_binary,
)
from .client import (
    DiagnosticInfo,
    Location,
    LspClient,
    LspError,
    LspTimeout,
    SymbolInfo,
    VhdlLsp,
    WorkspaceSymbolInfo,
    default_libraries_dir,
    path_to_uri,
    server_version,
)
from .veridian import VeridianLsp

__all__ = [
    "MODE_FALLBACK",
    "MODE_LSP",
    "AnalyzerStatus",
    "DiagnosticInfo",
    "Location",
    "LspClient",
    "LspError",
    "LspTimeout",
    "SymbolInfo",
    "VeridianLsp",
    "VhdlLsp",
    "WorkspaceSymbolInfo",
    "analyzer_status",
    "build_analyzer_statuses",
    "default_libraries_dir",
    "path_to_uri",
    "resolve_binary",
    "server_version",
]
