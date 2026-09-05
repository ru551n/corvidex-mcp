"""Tests for documentation chunking (Markdown, reST, plain text)."""

from __future__ import annotations

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.indexing.common import extract_code_identifiers
from corvidex_mcp.indexing.docs import chunk_doc_file
from corvidex_mcp.models import CollectionName, ContentType

CFG = RepositoryConfig(name="docs", url="git@example.com:co/docs.git")

MARKDOWN_DOC = """\
# VHDL Coding Standard

General preamble paragraph without any structure at all.

## Reset conventions

All async resets are named rst_n and must be deasserted synchronously.

```vhdl
p_write : process (clk, rst_n) is
begin
  if rst_n = '0' then
    wr_ptr <= 0;
  end if;
end process;
```

## Naming

Signals use snake_case. The write pointer is wr_ptr.

### Deep section

Details of wr_ptr and rd_ptr usage.
"""

RST_DOC = """\
VHDL Coding Standard
====================

Preamble.

Reset conventions
-----------------

Async resets are named rst_n.

.. code-block:: vhdl

   p_write : process (clk, rst_n) is
   begin
     wr_ptr <= 0;
   end process;

Naming
------

Signals use snake_case, e.g. wr_ptr.
"""

TXT_DOC = (
    "Reset conventions.\n"
    "All async resets are named rst_n and must be deasserted "
    "synchronously with the clock domain they enter.\n"
    "\n"
    "Naming.\n"
    "Signals use snake_case, for example wr_ptr and rd_ptr, and "
    "constants use UPPER_CASE such as C_TIMEOUT for the timeout value.\n"
)


def test_markdown_sections():
    chunks = chunk_doc_file(CFG, "docs/standard.md", MARKDOWN_DOC, "abc123", "markdown")
    headings = [c.heading for c in chunks]
    assert "VHDL Coding Standard" in headings
    assert "Reset conventions" in headings
    assert "Naming" in headings
    assert "Deep section" in headings
    # The first-level preamble is its own section, bounded by ## Reset.
    top = next(c for c in chunks if c.heading == "VHDL Coding Standard")
    reset = next(c for c in chunks if c.heading == "Reset conventions")
    naming = next(c for c in chunks if c.heading == "Naming")
    assert top.end_line < reset.start_line
    assert reset.end_line < naming.start_line
    # The VHDL fence stays inside the chunk content...
    assert "p_write" in reset.content
    # ...and its identifiers land in the symbols field (cross-referencing).
    assert "rst_n" in reset.symbols
    assert "wr_ptr" in reset.symbols
    assert "p_write" in reset.symbols
    # Deeper headings are attributed to their top-level section.
    deep = next(c for c in chunks if c.heading == "Deep section")
    assert deep.section == "Naming"
    assert deep.start_line >= naming.start_line
    for chunk in chunks:
        assert chunk.content_type is ContentType.DOCUMENTATION
        assert chunk.collection is CollectionName.DOCS
        assert chunk.symbol_kind == "section"
        assert chunk.file == "docs/standard.md"
        assert chunk.branch == "main"
        assert chunk.commit == "abc123"


def test_markdown_fence_without_vhdl_symbols():
    doc = "# T\n\n```python\nprint(1)\n```\n\ntext\n"
    chunks = chunk_doc_file(CFG, "d.md", doc, "c1", "markdown")
    # A python fence still yields identifiers (print), but nothing else
    # leaks out.
    assert "print" in chunks[0].symbols


def test_rst_sections():
    chunks = chunk_doc_file(
        CFG, "docs/standard.rst", RST_DOC, "abc123", "restructuredtext"
    )
    headings = [c.heading for c in chunks]
    assert "VHDL Coding Standard" in headings
    assert "Reset conventions" in headings
    assert "Naming" in headings
    reset = next(c for c in chunks if c.heading == "Reset conventions")
    assert "rst_n" in reset.symbols
    assert "wr_ptr" in reset.symbols
    assert "p_write" in reset.symbols
    naming = next(c for c in chunks if c.heading == "Naming")
    assert reset.end_line < naming.start_line


def test_plain_text_paragraphs():
    chunks = chunk_doc_file(CFG, "notes.txt", TXT_DOC, "abc123", "text")
    assert len(chunks) >= 2
    first, second = chunks[0], chunks[1]
    assert first.symbol_kind == "paragraph"
    assert first.start_line == 1
    assert first.end_line < second.start_line
    assert "rst_n" in first.content
    assert "wr_ptr" in second.content
    # No code fences -> no symbols field content.
    assert all(c.symbols == () for c in chunks)


def test_short_file_single_paragraph():
    doc = "Short note about resets. They use rst_n.\n"
    chunks = chunk_doc_file(CFG, "tiny.txt", doc, "abc123", "text")
    assert len(chunks) == 1
    assert "rst_n" in chunks[0].content


def test_empty_file_no_chunks():
    assert chunk_doc_file(CFG, "e.md", "", "c1", "markdown") == []
    assert chunk_doc_file(CFG, "e.txt", "   \n\n", "c1", "text") == []


def test_long_paragraph_split():
    line = "x " * 300  # 600 chars/line
    doc = "\n".join([line] * 10) + "\n"  # ~6000 chars
    chunks = chunk_doc_file(CFG, "long.txt", doc, "c1", "text")
    assert len(chunks) > 1
    # Line ranges must be contiguous and cover the file.
    prev_end = 0
    for chunk in chunks:
        assert chunk.start_line > prev_end
        prev_end = chunk.end_line
    assert prev_end == 10


def test_markdown_no_headings_falls_back_to_paragraphs():
    doc = (
        "First paragraph that is definitely longer than forty characters "
        "of plain prose.\n\n"
        "Second paragraph also long enough to stand on its own as a "
        "separate chunk of text.\n"
    )
    chunks = chunk_doc_file(CFG, "n.md", doc, "c1", "markdown")
    assert len(chunks) == 2
    assert chunks[0].symbol_kind == "paragraph"
    assert chunks[1].start_line == 3


def test_extract_code_identifiers():
    code = "fifo_write(clk, rst_n, wr_ptr); // timeout: C_TIMEOUT\n"
    out = extract_code_identifiers(code)
    assert "fifo_write" in out
    assert "rst_n" in out
    assert "wr_ptr" in out
    assert "C_TIMEOUT" in out  # comments: identifier still extracted (fine)
    assert len(out) == len(set(out))


def test_extract_code_identifiers_caps_and_noise():
    code = " ".join(f"id{i:02d}" for i in range(150)) + " 123 a"
    from corvidex_mcp.indexing.common import MAX_SYMBOLS

    out = extract_code_identifiers(code)
    assert len(out) == MAX_SYMBOLS
    assert "123" not in out
    assert "a" not in out  # single-letter identifiers are noise
