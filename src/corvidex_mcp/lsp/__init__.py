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
from .libraries import (
    DEFAULT_LIBRARY,
    RESERVED_LIBRARY_NAMES,
    group_by_library,
    infer_library,
)
from .veridian import VeridianLsp

__all__ = [
    "DEFAULT_LIBRARY",
    "MODE_FALLBACK",
    "MODE_LSP",
    "RESERVED_LIBRARY_NAMES",
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
    "group_by_library",
    "infer_library",
    "path_to_uri",
    "resolve_binary",
    "server_version",
]
