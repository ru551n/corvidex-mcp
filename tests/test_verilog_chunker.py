"""Tests for Verilog/SystemVerilog semantic chunking.

Covers the Veridian LSP-symbol path (plain names, standard kinds,
anonymous tasks), the structural fallback (module/package/program +
always/function/task), the whole-file fallback, and identifier
extraction with comment/directive handling.
"""

from __future__ import annotations

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.indexing.verilog import (
    MIN_INNER_SPAN,
    chunk_verilog_file,
    extract_identifiers,
)
from corvidex_mcp.lsp.client import SymbolInfo
from corvidex_mcp.models import CollectionName, ContentType

CFG = RepositoryConfig(name="repo", url="git@example.com:co/repo.git")

FIFO_SV = """\
`include "common.svh"

module fifo #(
  parameter int DEPTH = 4
) (
  input  logic clk,
  input  logic rst,
  output logic [7:0] dout
);
  logic [7:0] r;

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      dout <= 8'h0;
    end else begin
      dout <= r;
    end
  end

  function int add8(input int a, input int b);
    int t;
    t = a + b;
    add8 = t;
    return add8;
  endfunction

  task automatic do_write(input logic [7:0] d);
    r <= d;
    dout <= d;
    dout <= d;
    dout <= d;
  endtask
endmodule
"""

PKG_SV = """\
package common;
  localparam int FIFO_DEPTH = 4;
  typedef logic [7:0] byte_t;
  function int sq(input int x);
    sq = x * x;
    return sq;
    return sq;
  endfunction
endpackage
"""

PROG_SV = """\
program tb;
  initial begin
    $display("hello %0d", 1);
    $display("world %0d", 2);
    $display("done %0d", 3);
    $finish;
  end
endprogram
"""


def _sym(name, kind, start, end, children=()):
    return SymbolInfo(name, kind, start, end, tuple(children))


def test_lsp_path_normalized_and_native_kinds():
    # Veridian-shaped tree: plain names, standard kinds.
    tree = (
        _sym(
            "fifo",
            2,
            2,
            32,
            (
                _sym("DEPTH", 26, 3, 3),
                _sym("clk", 7, 5, 5),
                _sym("r", 13, 9, 9),
                _sym("add8", 12, 19, 24),
                _sym("", 12, 26, 31),  # anonymous task
            ),
        ),
    )
    chunks = chunk_verilog_file(
        CFG, "rtl/fifo.sv", FIFO_SV, "abc123", "systemverilog", lsp_symbols=tree
    )
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert ("design_unit", "fifo") in by_symbol
    assert ("function", "add8") in by_symbol
    assert ("task", "do_write") in by_symbol
    module = by_symbol[("design_unit", "fifo")]
    assert module.native_symbol_kind == "module"
    assert module.module is None
    add8 = by_symbol[("function", "add8")]
    assert add8.native_symbol_kind == "function"
    assert add8.module == "fifo"
    task = by_symbol[("task", "do_write")]
    assert task.module == "fifo"
    for chunk in chunks:
        assert chunk.language == "systemverilog"
        assert chunk.collection is CollectionName.HDL
        assert chunk.content_type is ContentType.SOURCE


def test_lsp_path_program_and_package():
    tree = (
        _sym("common", 4, 0, 8),
        _sym("tb", 2, 10, 17),
    )
    content = PKG_SV + "\n" + PROG_SV
    chunks = chunk_verilog_file(
        CFG, "rtl/x.sv", content, "abc123", "systemverilog", lsp_symbols=tree
    )
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert ("package", "common") in by_symbol
    pkg = by_symbol[("package", "common")]
    assert pkg.native_symbol_kind == "package"
    prog = by_symbol[("design_unit", "tb")]
    assert prog.native_symbol_kind == "program"


def test_structural_fallback_full():
    chunks = chunk_verilog_file(
        CFG, "rtl/fifo.sv", FIFO_SV, "abc123", "systemverilog", lsp_symbols=None
    )
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert ("design_unit", "fifo") in by_symbol
    # always_ff -> process (native keeps the keyword).
    proc = next(
        c
        for c in chunks
        if c.symbol_kind == "process" and c.native_symbol_kind == "always_ff"
    )
    assert proc.module == "fifo"
    # function + task inners.
    assert ("function", "add8") in by_symbol
    assert ("task", "do_write") in by_symbol
    module = by_symbol[("design_unit", "fifo")]
    assert module.native_symbol_kind == "module"


def test_structural_fallback_package():
    chunks = chunk_verilog_file(
        CFG, "rtl/common.sv", PKG_SV, "abc123", "systemverilog", lsp_symbols=None
    )
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    assert ("package", "common") in by_symbol
    # The inner function (>= MIN_INNER_SPAN) earns its own chunk.
    assert ("function", "sq") in by_symbol


def test_structural_fallback_program():
    chunks = chunk_verilog_file(
        CFG, "tb.sv", PROG_SV, "abc123", "systemverilog", lsp_symbols=None
    )
    by_symbol = {(c.symbol_kind, c.symbol): c for c in chunks}
    prog = by_symbol[("design_unit", "tb")]
    assert prog.native_symbol_kind == "program"


def test_whole_file_fallback_when_no_construct():
    chunks = chunk_verilog_file(
        CFG,
        "rtl/none.sv",
        "// just a comment\n\n  // nothing here\n",
        "abc123",
        "verilog",
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol_kind == "file"
    assert chunk.native_symbol_kind == "file"
    assert chunk.symbol == "none"


def test_broken_file_extends_unit_to_eof():
    chunks = chunk_verilog_file(
        CFG, "rtl/bad.sv", "module broken (\n  input clk\n", "abc123", "verilog"
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert (chunk.symbol_kind, chunk.symbol) == ("design_unit", "broken")
    assert chunk.end_line == 2


def test_identifiers_strip_comments_and_directives():
    ident = extract_identifiers(
        'module m; `include "x" /* c */ // l\nlogic FIFO_DEPTH; endmodule'
    )
    assert "FIFO_DEPTH" in ident
    assert "m" in ident
    # block-comment content and the `include directive name are dropped.
    assert "c" not in ident
    assert "include" not in ident


def test_identifiers_multiline_block_comment():
    text = "/* start\nFIFO_DEPTH\nmid */ module m; endmodule\n"
    ident = extract_identifiers(text)
    assert "FIFO_DEPTH" not in ident
    assert "m" in ident


def test_identifiers_dedup_and_cap():
    text = "a a a b b c\n" * 50
    ident = extract_identifiers(text)
    assert len(set(ident)) == len(ident)
    assert len(ident) <= 100


def test_min_inner_span_constant():
    assert MIN_INNER_SPAN == 5
