"""Tests for VHDL semantic chunking (LSP path, structural fallback, assembly)."""

from __future__ import annotations

import itertools

import pytest

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.indexing.vhdl import (
    MAX_SYMBOLS,
    ChunkSpec,
    chunk_vhdl_file,
    extract_identifiers,
)
from corvidex_mcp.lsp.client import SymbolInfo
from corvidex_mcp.models import CollectionName, ContentType

FIFO_VHDL = """\
library ieee;
use ieee.std_logic_1164.all;

entity fifo is
  port (
    clk : in  std_logic;
    rst : in  std_logic
  );
end entity fifo;

architecture rtl of fifo is
  signal wr_ptr : integer := 0;
begin
  p_write : process (clk, rst) is
  begin
    if rst = '1' then
      wr_ptr <= 0;
    end if;
  end process p_write;
  p_read : process (clk) is
  begin
    wr_ptr <= wr_ptr + 1;
  end process p_read;
end architecture rtl;
"""

PKG_VHDL = """\
package fifo_pkg is
  type mem_array is array (0 to 7) of std_ulogic;
end package fifo_pkg;
package body fifo_pkg is
  function to_int(x : std_ulogic_vector) return integer is
    variable acc : integer;
  begin
    acc := 0;
    return acc;
  end function to_int;
end package body fifo_pkg;
"""

CFG = RepositoryConfig(name="repo", url="git@example.com:co/repo.git")


def _lines() -> list[str]:
    return FIFO_VHDL.splitlines()


def test_structural_fallback_specs():
    chunks = chunk_vhdl_file(CFG, "rtl/fifo.vhd", FIFO_VHDL, "abc123", lsp_symbols=None)
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    # entity + architecture + the larger process; p_read (4 lines) stays
    # inside the architecture chunk.
    assert set(by_symbol) == {
        ("entity", "fifo"),
        ("architecture", "rtl"),
        ("process", "p_write"),
    }
    ent = by_symbol[("entity", "fifo")]
    assert (ent.start_line, ent.end_line) == (4, 9)
    assert ent.entity == "fifo"
    assert ent.architecture is None
    arch = by_symbol[("architecture", "rtl")]
    assert (arch.start_line, arch.end_line) == (11, 24)
    assert arch.entity == "fifo"
    assert arch.architecture == "rtl"
    proc = by_symbol[("process", "p_write")]
    assert (proc.start_line, proc.end_line) == (14, 19)
    assert proc.entity == "fifo"
    assert proc.architecture == "rtl"
    # All chunks share the file's library and repository attribution.
    for chunk in chunks:
        assert chunk.library == "ieee"
        assert chunk.repository == "repo"
        assert chunk.branch == "main"
        assert chunk.commit == "abc123"
        assert chunk.content_type is ContentType.SOURCE
        assert chunk.collection is CollectionName.HDL
        assert chunk.language == "vhdl"
        assert chunk.file == "rtl/fifo.vhd"


def test_chunk_content_is_exact_slice():
    chunks = chunk_vhdl_file(CFG, "rtl/fifo.vhd", FIFO_VHDL, "abc123")
    lines = _lines()
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["p_write"].content == "\n".join(lines[13:19])
    assert by_symbol["p_write"].content.lstrip().startswith("p_write : process")
    assert by_symbol["p_write"].content.endswith("end process p_write;")
    # Construct names and referenced identifiers land in the symbols list.
    assert "p_write" in by_symbol["p_write"].symbols
    assert "wr_ptr" in by_symbol["p_write"].symbols
    assert "fifo" in by_symbol["fifo"].symbols


def test_lsp_path_symbols():
    """The vhdl_ls shape: entity and architecture are top-level siblings."""
    syms = (
        SymbolInfo(
            name="entity 'fifo'",
            kind=2,
            start_line=0,
            end_line=5,
            children=(),
        ),
        SymbolInfo(
            name="architecture 'rtl'",
            kind=2,
            start_line=6,
            end_line=15,
            children=(
                SymbolInfo(
                    name="process 'p_write'",
                    kind=3,
                    start_line=9,
                    end_line=14,
                    children=(),
                ),
                # Too small to earn its own chunk (span 3 < MIN_INNER_SPAN).
                SymbolInfo(
                    name="process 'p_small'",
                    kind=3,
                    start_line=12,
                    end_line=14,
                    children=(),
                ),
            ),
        ),
    )
    chunks = chunk_vhdl_file(CFG, "rtl/fifo.vhd", FIFO_VHDL, "abc123", lsp_symbols=syms)
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert set(by_symbol) == {
        ("entity", "fifo"),
        ("architecture", "rtl"),
        ("process", "p_write"),
    }
    arch = by_symbol[("architecture", "rtl")]
    assert (arch.start_line, arch.end_line) == (7, 16)
    assert arch.entity == "fifo"  # from the "architecture rtl of fifo is" line
    proc = by_symbol[("process", "p_write")]
    assert (proc.start_line, proc.end_line) == (10, 15)
    assert proc.entity == "fifo"
    assert proc.architecture == "rtl"


def test_lsp_entity_context_backfill():
    """An architecture whose declaration line lacks 'of' still gets the
    sibling entity's name."""
    lines = [
        "entity solo is",
        "end entity solo;",
        "architecture only is",
        "begin",
        "end architecture only;",
    ]
    syms = (
        SymbolInfo("entity 'solo'", 2, 0, 1),
        SymbolInfo("architecture 'only'", 2, 2, 4),
    )
    chunks = chunk_vhdl_file(
        CFG, "rtl/solo.vhd", "\n".join(lines), "c1", lsp_symbols=syms
    )
    arch = next(c for c in chunks if c.symbol_kind == "architecture")
    assert arch.entity == "solo"


def test_lsp_prefix_absent_names():
    """Servers that do not kind-prefix names still resolve via kind ints."""
    syms = (SymbolInfo("fifo", 4, 0, 4),)  # package (kind 4), no prefix
    text = "package fifo is\n  constant C : integer := 1;\nend package fifo;\n"
    chunks = chunk_vhdl_file(CFG, "rtl/pkg.vhd", text, "c1", lsp_symbols=syms)
    assert len(chunks) == 1
    assert chunks[0].symbol_kind == "package"
    assert chunks[0].symbol == "fifo"


def test_package_and_function_fallback():
    chunks = chunk_vhdl_file(CFG, "rtl/fifo_pkg.vhd", PKG_VHDL, "abc123")
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert set(by_symbol) == {
        ("package", "fifo_pkg"),
        ("package_body", "fifo_pkg"),
        ("function", "to_int"),
    }
    pkg = by_symbol[("package", "fifo_pkg")]
    assert (pkg.start_line, pkg.end_line) == (1, 3)
    body = by_symbol[("package_body", "fifo_pkg")]
    assert (body.start_line, body.end_line) == (4, 11)
    func = by_symbol[("function", "to_int")]
    assert (func.start_line, func.end_line) == (5, 10)


def test_unparseable_file_becomes_one_chunk():
    content = "-- just a note\nx <= 1;\n"
    chunks = chunk_vhdl_file(CFG, "rtl/mystery.vhd", content, "abc123")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol == "mystery"
    assert chunk.symbol_kind == "file"
    assert (chunk.start_line, chunk.end_line) == (1, 2)
    assert chunk.content == content.strip()


def test_extract_identifiers():
    content = (
        "entity fifo is  -- port (clk, rst)\n"
        "  port (clk : in std_logic);\n"
        "  signal mem : std_ulogic_vector(7 downto 0);\n"
        "end entity;\n"
    )
    out = extract_identifiers(content)
    # Keywords (entity, port, in, signal, end) are dropped; the comment's
    # identifiers are ignored; order is first-occurrence.
    assert out == ("fifo", "clk", "std_logic", "mem", "std_ulogic_vector")


def test_extract_identifiers_capped():
    content = "\n".join(f"x{i} <= y{i};" for i in range(MAX_SYMBOLS + 50))
    out = extract_identifiers(content)
    assert len(out) == MAX_SYMBOLS
    assert len(set(out)) == MAX_SYMBOLS


def test_chunk_order_follows_source():
    chunks = chunk_vhdl_file(CFG, "rtl/fifo.vhd", FIFO_VHDL, "abc123")
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)


def test_canonical_ids_distinct_for_same_process_label():
    """The same process label in two architectures must not collide."""
    text = "\n".join(
        [
            "entity t is end entity t;",
            "architecture a1 of t is",
            "begin",
            "  p : process is",
            "  begin",
            "    wait;",
            "    wait;",
            "  end process;",
            "end architecture a1;",
            "architecture a2 of t is",
            "begin",
            "  p : process is",
            "  begin",
            "    wait;",
            "    wait;",
            "  end process;",
            "end architecture a2;",
        ]
    )
    chunks = chunk_vhdl_file(CFG, "rtl/t.vhd", text, "abc123")
    ids = {c.canonical_id for c in chunks}
    assert len(ids) == len(chunks)
    procs = [c for c in chunks if c.symbol_kind == "process" and c.symbol == "p"]
    assert len(procs) == 2
    assert procs[0].architecture == "a1"
    assert procs[1].architecture == "a2"


def test_empty_file_raises():
    with pytest.raises(ValueError, match="empty VHDL file"):
        chunk_vhdl_file(CFG, "rtl/empty.vhd", "", "abc123")


def test_spec_is_frozen():
    from dataclasses import FrozenInstanceError

    spec = ChunkSpec("x", "entity", 1, 2)
    with pytest.raises(FrozenInstanceError):
        spec.symbol = "y"  # type: ignore[misc]


# -- size bound + structural split -------------------------------------------

from corvidex_mcp.indexing.common import MAX_CONTENT_CHARS  # noqa: E402


def _big_architecture(n_proc: int = 16, decl_lines: int = 80) -> tuple[str, int]:
    """An architecture far larger than MAX_CONTENT_CHARS: a long
    declaration part and n_proc sizeable processes. Returns (content,
    1-based begin line)."""
    decl = "\n".join(
        f"  signal sig_{i:03d} : std_logic_vector(31 downto 0) := '0' & '1'; -- d{i}"
        for i in range(decl_lines)
    )
    procs = []
    for p in range(n_proc):
        body = "\n".join(
            f"      reg_{p}_{i} <= data_{p}_{i} & sig_{p % decl_lines}; -- s{p}.{i}"
            for i in range(20)
        )
        procs.append(
            f"  p_proc_{p} : process (clk, rst)\n"
            f"  begin\n"
            f"{body}\n"
            f"  end process p_proc_{p};"
        )
    content = (
        "architecture big of top is\n"
        f"{decl}\n"
        "begin\n" + "\n".join(procs) + "\n"
        "end architecture big;\n"
    )
    return content, 2 + decl_lines


def test_small_constructs_are_single_chunks():
    # The canonical FIFO file is far below the bound: unchanged behaviour
    # (one chunk per spec, exact line ranges).
    chunks = chunk_vhdl_file(CFG, "rtl/fifo.vhd", FIFO_VHDL, "abc123")
    assert len(chunks) == 3
    arch = next(c for c in chunks if c.symbol_kind == "architecture")
    assert (arch.start_line, arch.end_line) == (11, 24)


def test_oversized_architecture_splits_along_its_structure():
    content, begin_line = _big_architecture()
    assert len(content) > MAX_CONTENT_CHARS
    chunks = chunk_vhdl_file(CFG, "rtl/big.vhd", content, "abc123")
    arch = [c for c in chunks if c.symbol_kind == "architecture" and c.symbol == "big"]
    # Split into several parts, all keeping the construct's identity.
    assert len(arch) >= 3
    # Every part's content respects the bound (no pathological single line
    # here, so the bound holds strictly).
    assert all(len(c.content) <= MAX_CONTENT_CHARS for c in chunks)
    # The parts tile the construct's line range exactly: contiguous,
    # in order, no gaps, no overlaps.
    arch.sort(key=lambda c: c.start_line)
    assert arch[0].start_line == 1
    assert arch[-1].end_line == len(content.splitlines())
    for prev, cur in itertools.pairwise(arch):
        assert cur.start_line == prev.end_line + 1
    # The first part is the declaration part (ends before begin); the
    # second part starts with the begin line.
    assert arch[0].end_line == begin_line - 1
    assert arch[1].content.splitlines()[0].strip() == "begin"
    # The closing end line sits in the last part.
    assert arch[-1].content.splitlines()[-1].startswith("end architecture")


def test_oversized_architecture_statement_windows():
    # A single process larger than the bound must still be windowed:
    # every line stays in the index, and the bound holds.
    body = "\n".join(
        f"      sig_{i:04d} <= sig_{(i + 1) % 400} & '1';  -- filler {i}"
        for i in range(400)
    )
    content = (
        "architecture big of top is\n"
        "  signal seed : std_logic;\n"
        "begin\n"
        "  p_huge : process (clk) is\n"
        "  begin\n"
        f"{body}\n"
        "  end process p_huge;\n"
        "end architecture big;\n"
    )
    assert len(content) > MAX_CONTENT_CHARS
    chunks = chunk_vhdl_file(CFG, "rtl/huge.vhd", content, "abc123")
    lines = content.splitlines()
    assert len(content) > MAX_CONTENT_CHARS
    # Coverage: the architecture parts tile the construct range.
    arch = sorted((c for c in chunks if c.symbol == "big"), key=lambda c: c.start_line)
    assert arch[0].start_line == 1
    assert arch[-1].end_line == len(lines)
    for prev, cur in itertools.pairwise(arch):
        assert cur.start_line == prev.end_line + 1
    # The oversized process also gets its own (split) chunks from its own
    # spec.
    huge = sorted(
        (c for c in chunks if c.symbol == "p_huge"), key=lambda c: c.start_line
    )
    assert len(huge) >= 2
    assert all(len(c.content) <= MAX_CONTENT_CHARS for c in huge)


def test_oversized_entity_windows_on_blank_lines():
    # An entity with a huge port list (no begin/end structure) is windowed
    # generically, preferring blank-line breaks.
    ports = ",\n".join(
        f"    port_{i:03d} : in std_logic_vector(31 downto 0)" for i in range(300)
    )
    content = f"entity wide is\n  port (\n{ports}\n  );\nend entity wide;\n"
    assert len(content) > MAX_CONTENT_CHARS
    chunks = chunk_vhdl_file(CFG, "rtl/wide.vhd", content, "abc123")
    ents = sorted((c for c in chunks if c.symbol == "wide"), key=lambda c: c.start_line)
    assert len(ents) >= 2
    assert all(len(c.content) <= MAX_CONTENT_CHARS for c in ents)
    # Exact tiling of the entity range.
    assert ents[0].start_line == 1
    assert ents[-1].end_line == len(content.splitlines())
    for prev, cur in itertools.pairwise(ents):
        assert cur.start_line == prev.end_line + 1


def test_split_is_deterministic_and_ids_distinct():
    content, _begin = _big_architecture()
    a = chunk_vhdl_file(CFG, "rtl/big.vhd", content, "abc123")
    b = chunk_vhdl_file(CFG, "rtl/big.vhd", content, "abc123")
    assert [c.canonical_id for c in a] == [c.canonical_id for c in b]
    assert len({c.canonical_id for c in a}) == len(a)
