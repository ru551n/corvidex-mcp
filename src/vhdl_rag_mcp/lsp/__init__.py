"""LSP client for vhdl_ls (see :mod:`vhdl_rag_mcp.lsp.client`)."""

from .client import (
    DiagnosticInfo,
    LspError,
    LspTimeout,
    SymbolInfo,
    VhdlLsp,
    default_libraries_dir,
    path_to_uri,
)

__all__ = [
    "DiagnosticInfo",
    "LspError",
    "LspTimeout",
    "SymbolInfo",
    "VhdlLsp",
    "default_libraries_dir",
    "path_to_uri",
]
