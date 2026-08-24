"""FastMCP server exposing VHDL RAG search over configured Git repositories."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "vhdl_rag_mcp",
    instructions=(
        "Semantic search over an organization's VHDL code, VHDL-related "
        "documentation, and general source code. Use search_vhdl for "
        "reference VHDL implementations (entities, architectures, processes, "
        "packages, functions, reset/clock/FSM patterns), search_docs for "
        "standards and design documentation, search_code for general "
        "C/C++/Python code, and search_knowledge when the answer may span "
        "any of them. Use get_source for the full text of a known file. "
        "Every result carries exact source attribution (repository, file, "
        "line range, commit) and cross-referenced related chunks. Use "
        "repository_status to see what is indexed; sync_repositories to "
        "force an update."
    ),
)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
