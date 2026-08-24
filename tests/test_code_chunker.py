"""Tests for general source-code chunking (C/C++ and Python)."""

from __future__ import annotations

from vhdl_rag_mcp.config import RepositoryConfig
from vhdl_rag_mcp.indexing.code import chunk_code_file
from vhdl_rag_mcp.models import CollectionName, ContentType

CFG = RepositoryConfig(name="code", url="git@example.com:co/code.git")

C_FILE = """\
#include <stdio.h>

/* a header comment
   spanning lines */
static int helper(int x) {
    return x + 1;
}

int fifo_write(int *mem, int ptr, int val) {
    mem[ptr] = val;
    return 0;
}

int main(void) {
    int a = 5; // trailing comment { unbalanced
    printf("%d { }\\n", a);
    return helper(a);
}
"""

PY_FILE = """\
import os


@decorator_a
@decorator_b
def fifo_write(mem, ptr):
    x = 1
    return x


class Fifo:
    depth = 8

    def read(self):
        return 0


module_level = "module scope"
another = module_level + 1
"""

HEADER_FILE = """\
#pragma once
#define FIFO_DEPTH 8
extern int fifo_write(int *mem, int ptr, int val);
"""


def test_c_functions():
    chunks = chunk_code_file(CFG, "src/fifo.c", C_FILE, "abc123", "c")
    by_name = {c.symbol: c for c in chunks}
    assert set(by_name) == {"helper", "fifo_write", "main"}
    helper = by_name["helper"]
    assert (helper.start_line, helper.end_line) == (5, 7)
    fw = by_name["fifo_write"]
    assert (fw.start_line, fw.end_line) == (9, 12)
    assert fw.content.startswith("int fifo_write")
    assert fw.content.endswith("}")
    main = by_name["main"]
    assert (main.start_line, main.end_line) == (14, 18)
    # Comment/string braces must not confuse the scanner: the exact
    # (14, 18) range above only holds if they don't, and the chunk keeps
    # the raw source (comment and string included).
    assert "// trailing comment" in main.content
    assert main.content.endswith("}")
    # Cross-referencing identifiers.
    assert "fifo_write" in fw.symbols
    assert "mem" in fw.symbols
    for chunk in chunks:
        assert chunk.content_type is ContentType.CODE
        assert chunk.collection is CollectionName.CODE
        assert chunk.symbol_kind == "function"
        assert chunk.language == "c"
        assert chunk.file == "src/fifo.c"
        assert chunk.commit == "abc123"


def test_python_units():
    chunks = chunk_code_file(CFG, "src/fifo.py", PY_FILE, "abc123", "python")
    by_name = {c.symbol: c for c in chunks}
    assert set(by_name) == {"fifo_write", "Fifo", "fifo"}
    fw = by_name["fifo_write"]
    assert fw.symbol_kind == "function"
    # Decorators are included in the chunk.
    assert fw.content.startswith("@decorator_a")
    cls = by_name["Fifo"]
    assert cls.symbol_kind == "class"
    # Nested def is inside the class chunk, not its own top-level chunk.
    assert "def read" in cls.content
    # Module-level statements without a def/class form a file-scope
    # gap unit (the single-line import is dropped as noise).
    assert "module_level" in by_name["fifo"].content
    assert "another" in by_name["fifo"].content
    assert "import" not in by_name["fifo"].content
    for chunk in chunks:
        assert chunk.collection is CollectionName.CODE
        assert chunk.language == "python"


def test_python_no_units_fallback():
    content = "CONSTANT = 42\nother = 1\n"
    chunks = chunk_code_file(CFG, "cfg.py", content, "c1", "python")
    assert len(chunks) == 1
    assert chunks[0].symbol == "cfg"
    assert chunks[0].symbol_kind == "file"


def test_header_without_bodies_falls_back_to_file():
    chunks = chunk_code_file(CFG, "inc/fifo.h", HEADER_FILE, "c1", "c")
    assert len(chunks) == 1
    assert chunks[0].symbol == "fifo"
    assert chunks[0].symbol_kind == "file"
    assert "fifo_write" in chunks[0].symbols


def test_empty_file():
    assert chunk_code_file(CFG, "e.c", "", "c1", "c") == []
    assert chunk_code_file(CFG, "e.py", "   \n", "c1", "python") == []


def test_cpp_pointer_reference_syntax():
    content = (
        "void reset_fsm(uint8_t *state, const cfg_t &cfg) {\n"
        "  *state = 0;\n"
        "}\n"
        "void step_fsm(uint8_t *state) { *state += 1; }\n"
    )
    chunks = chunk_code_file(CFG, "fsm.cpp", content, "c1", "cpp")
    names = {c.symbol for c in chunks}
    assert names == {"reset_fsm", "step_fsm"}
    for c in chunks:
        assert c.language == "cpp"


def test_cpp_no_unbalanced_braces_in_strings():
    content = 'const char *msg = "brace { here";\n\nint f(void) {\n  return 0;\n}\n'
    chunks = chunk_code_file(CFG, "s.cpp", content, "c1", "cpp")
    names = {c.symbol for c in chunks}
    assert names == {"f"}
