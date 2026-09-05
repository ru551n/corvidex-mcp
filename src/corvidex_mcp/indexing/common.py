"""Shared helpers for the per-domain chunkers."""

from __future__ import annotations

import re

#: Upper bound on chunk content size (chars). Well within the embedding
#: models' 8192-token context, and small enough to stay useful as a
#: RAG unit.
MAX_CONTENT_CHARS = 12000
#: Identifier cap per chunk (payload size bound).
MAX_SYMBOLS = 100

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: Dotted identifiers like ``foo.bar`` are kept whole; pure numbers and
#: single letters are dropped (noise for cross-referencing).


def extract_code_identifiers(code: str) -> tuple[str, ...]:
    """Identifiers referenced/defined in a code snippet.

    Language-agnostic: first-occurrence order, deduplicated, capped at
    MAX_SYMBOLS. Comments are ignored (``//``, ``#``, ``--`` prefixes at
    token start). Used for documentation code fences and source-code
    chunks — the cross-referencing key that ties documentation to VHDL
    and code.
    """
    seen: dict[str, None] = {}
    for raw in _IDENT_RE.findall(code):
        ident = raw.rstrip(".")
        if len(ident) < 2 or ident.isdigit():
            continue
        if ident not in seen:
            seen[ident] = None
            if len(seen) >= MAX_SYMBOLS:
                break
    return tuple(seen)
