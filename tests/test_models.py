"""Tests for the normalized HDL semantic model (chunk + result rendering)."""

from __future__ import annotations

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.models import (
    INDEX_SCHEMA_VERSION,
    Chunk,
    CollectionName,
    ContentType,
    SearchResult,
)


def make_chunk(**overrides: object) -> Chunk:
    base: dict[str, object] = {
        "repository": "repo",
        "branch": "main",
        "commit": "a" * 40,
        "file": "rtl/fifo.vhd",
        "content_type": ContentType.SOURCE,
        "language": "vhdl",
        "collection": CollectionName.HDL,
        "symbol": "fifo",
        "symbol_kind": "design_unit",
        "native_symbol_kind": "entity",
        "start_line": 1,
        "end_line": 10,
        "content": "entity fifo is end fifo;",
    }
    base.update(overrides)
    return Chunk(**base)  # type: ignore[arg-type]


def test_chunk_payload_carries_normalized_and_native_kinds() -> None:
    chunk = make_chunk()
    payload = chunk.payload()
    assert payload["symbol_kind"] == "design_unit"
    assert payload["native_symbol_kind"] == "entity"
    assert payload["collection"] == "hdl"
    assert payload["language"] == "vhdl"
    # Absent optional context is not stored.
    assert "module" not in payload
    assert "entity" not in payload


def test_chunk_payload_carries_module_context() -> None:
    chunk = make_chunk(
        file="rtl/fifo.sv",
        language="systemverilog",
        symbol="add8",
        symbol_kind="function",
        native_symbol_kind="function",
        module="fifo",
    )
    payload = chunk.payload()
    assert payload["module"] == "fifo"
    assert payload["native_symbol_kind"] == "function"


def test_canonical_id_stable_across_commits() -> None:
    chunk = make_chunk()
    other = make_chunk(commit="b" * 40)
    assert chunk.canonical_id == other.canonical_id


def test_result_render_hdl_fence_uses_language() -> None:
    result = SearchResult(
        result_type="hdl",
        repository="repo",
        commit="a" * 40,
        file="rtl/fifo.vhd",
        content="entity fifo is end fifo;",
        score=0.5,
        language="vhdl",
        symbol="fifo",
        symbol_kind="design_unit",
        native_symbol_kind="entity",
    )
    text = result.render()
    assert "```vhdl" in text
    # The native kind is shown next to the normalized one.
    assert "design_unit fifo (entity)" in text


def test_result_render_verilog_fence_and_module() -> None:
    result = SearchResult(
        result_type="hdl",
        repository="repo",
        commit="a" * 40,
        file="rtl/fifo.sv",
        content="always_ff @(posedge clk) begin end",
        score=0.5,
        language="systemverilog",
        symbol="write",
        symbol_kind="process",
        native_symbol_kind="always_ff",
        module="fifo",
    )
    text = result.render()
    assert "```systemverilog" in text
    assert "process write (always_ff)" in text
    assert "module fifo" in text


def test_result_render_code_fence_unchanged() -> None:
    result = SearchResult(
        result_type="code",
        repository="repo",
        commit="a" * 40,
        file="src/fifo.c",
        content="int fifo_write(void) { return 0; }",
        score=0.5,
        language="c",
        symbol="fifo_write",
        symbol_kind="function",
    )
    text = result.render()
    assert "```code" in text
    # No native kind: no parentheses.
    assert "function fifo_write)" not in text


def test_index_schema_version_is_explicit() -> None:
    assert INDEX_SCHEMA_VERSION == 2


def test_default_repository_domain_is_hdl() -> None:
    repo = RepositoryConfig(name="r", url="u")
    assert CollectionName.HDL in repo.domains
    # The legacy spelling is accepted as an alias.
    legacy = RepositoryConfig(name="r", url="u", domains=["vhdl", "docs"])
    assert legacy.domains == [CollectionName.HDL, CollectionName.DOCS]
    explicit = RepositoryConfig(name="r", url="u", domains=["hdl", "docs"])
    assert explicit.domains == [CollectionName.HDL, CollectionName.DOCS]
