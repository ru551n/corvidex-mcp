"""Tests for the normalized HDL semantic model (chunk + result rendering)."""

from __future__ import annotations

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.models import (
    INDEX_SCHEMA_VERSION,
    MAX_BODY_LINES,
    Chunk,
    CollectionName,
    ContentType,
    SearchResult,
    SearchResults,
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


def _long_result(lines: int, start_line: int = 595) -> SearchResult:
    """An hdl result whose chunk is ``lines`` lines long."""
    return SearchResult(
        result_type="hdl",
        repository="repo",
        commit="a" * 40,
        file="rtl/fifo.vhd",
        content="\n".join(f"signal s{i} : std_logic;" for i in range(lines)),
        score=0.5,
        language="vhdl",
        symbol="rtl",
        symbol_kind="architecture",
        start_line=start_line,
        end_line=start_line + lines - 1,
    )


def test_result_render_caps_the_body_and_names_the_follow_up_call() -> None:
    text = _long_result(200).render()
    body = text.split("```vhdl\n", 1)[1]
    quoted = [line for line in body.splitlines() if " | " in line]
    assert len(quoted) == MAX_BODY_LINES
    # The head is kept (that is where entity/process headers live).
    assert "signal s0 : std_logic;" in text
    assert "signal s199 : std_logic;" not in text
    # The marker is a ready-to-run call on this result's own lines:
    # 200 lines from 595, 40 quoted, so 635-794 remain.
    assert (
        "… 160 more lines — "
        'get_source("repo", "rtl/fifo.vhd", 635, 794) for the full text'
    ) in text


def test_result_render_does_not_elide_a_short_body() -> None:
    text = _long_result(MAX_BODY_LINES).render()
    assert "more lines" not in text
    assert f"signal s{MAX_BODY_LINES - 1} : std_logic;" in text


def test_result_render_numbers_body_lines_from_the_file_start() -> None:
    result = SearchResult(
        result_type="hdl",
        repository="repo",
        commit="a" * 40,
        file="rtl/fifo.vhd",
        content="entity fifo is\n  port (clk : in std_logic);\nend entity fifo;",
        score=0.5,
        language="vhdl",
        start_line=97,
        end_line=99,
    )
    text = result.render()
    assert "  97 | entity fifo is" in text
    assert "  98 |   port (clk : in std_logic);" in text
    assert "  99 | end entity fifo;" in text


def test_result_render_without_lines_has_no_gutter() -> None:
    result = SearchResult(
        result_type="docs",
        repository="repo",
        commit="a" * 40,
        file="docs/standard.md",
        content="Async resets are named rst_n.",
        score=0.5,
    )
    assert "| Async resets" not in result.render()
    assert "Async resets are named rst_n." in result.render()


def test_search_results_defaults_are_conservative() -> None:
    results = SearchResults([_long_result(1)])
    assert len(results) == 1
    assert not results.has_more
    assert not results.calibrated_scores


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
