"""Deterministic query expansion for RTL/HDL domain terminology.

A static synonym table, applied by appending matched aliases to the
query text before it is embedded/searched (never replacing the
original terms), so lexical (FTS5) and semantic (dense) legs both gain
recall for the many naming conventions a query can use for the same
concept (e.g. "clock" vs "clk", VHDL "generic" vs Verilog/SV
"parameter"). Fully static and offline: no model, no network call, so
it stays deterministic for the CI quality gate.
"""

from __future__ import annotations

import re

#: term -> aliases to append when the term appears in the query.
#: Keys and values are matched/inserted case-insensitively; extend as
#: real query gaps are found (see tests/quality/queries.json).
LEXICON: dict[str, tuple[str, ...]] = {
    "reset": ("rst", "rst_n", "async_reset", "areset"),
    "clock": ("clk",),
    "clock enable": ("clk_en", "ce"),
    "asynchronous": ("async",),
    "synchronous": ("sync",),
    "handshake": ("valid", "ready"),
    "flip-flop": ("ff", "register"),
    "flip flop": ("ff", "register"),
    "generic": ("parameter",),
    "parameter": ("generic",),
    "testbench": ("tb", "test_bench"),
    "fifo": ("queue", "buffer"),
    "state machine": ("fsm",),
    "fsm": ("state machine",),
    "signal": ("wire", "net"),
}

#: Safety cap: at most this many extra terms are appended, regardless
#: of how many lexicon entries match (keeps the FTS query and the
#: embedded text from being diluted by a term-heavy query).
MAX_EXTRA_TERMS = 6

#: Queries that look like an exact identifier lookup (contain an
#: underscore and no whitespace, e.g. "rst_n" or "fifo_write") skip
#: expansion: expanding a signal/symbol name works against exact-match
#: intent rather than for it. A plain single word (e.g. "clock") is
#: not an identifier lookup and is still expanded.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*$")


def expand_query(query: str) -> str:
    """Append matched lexicon synonyms to ``query``.

    The original query is always returned first and unmodified;
    matched aliases are appended space-separated. Returns ``query``
    unchanged when it looks like an exact identifier lookup, when
    nothing matches, or when ``query`` is empty/whitespace.
    """
    stripped = query.strip()
    if not stripped or _IDENTIFIER_RE.match(stripped):
        return query
    lowered = stripped.lower()
    extra: list[str] = []
    seen: set[str] = set()
    for term, aliases in LEXICON.items():
        if len(extra) >= MAX_EXTRA_TERMS:
            break
        if not re.search(rf"\b{re.escape(term)}\b", lowered):
            continue
        for alias in aliases:
            if len(extra) >= MAX_EXTRA_TERMS:
                break
            if alias in seen or alias == term:
                continue
            # Do not append an alias already present verbatim in the query.
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                continue
            seen.add(alias)
            extra.append(alias)
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"
