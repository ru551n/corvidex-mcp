"""LSP clients over stdio (see :mod:`vhdl_rag_mcp.lsp.client`)."""

from .analyzers import (
    MODE_FALLBACK,
    MODE_LSP,
    AnalyzerStatus,
    analyzer_status,
    build_analyzer_statuses,
    build_client,
    resolve_binary,
)
from .client import (
    DiagnosticInfo,
    LspClient,
    LspError,
    LspTimeout,
    SymbolInfo,
    VhdlLsp,
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
    "LspClient",
    "LspError",
    "LspTimeout",
    "SymbolInfo",
    "VeridianLsp",
    "VhdlLsp",
    "analyzer_status",
    "build_analyzer_statuses",
    "build_client",
    "default_libraries_dir",
    "path_to_uri",
    "resolve_binary",
    "server_version",
]
