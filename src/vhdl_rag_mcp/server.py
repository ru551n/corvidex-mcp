"""FastMCP server exposing VHDL RAG search over configured Git repositories."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "vhdl_rag_mcp",
    instructions=(
        "Semantic search over an organization's VHDL code and VHDL-related "
        "documentation. Use search_vhdl for reference VHDL implementations, "
        "search_docs for coding standards and design documentation, and "
        "search_knowledge when the answer may span both. Every result "
        "carries exact source attribution (repository, file, line range). "
        "Use repository_status to see what is indexed and when it was last "
        "synchronized."
    ),
)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
