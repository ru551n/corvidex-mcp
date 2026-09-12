"""Shared typed models for HDL, documentation, and code chunks.

HDL covers VHDL, Verilog, and SystemVerilog (one collection).

The :class:`Chunk` model is the single currency between the indexers, the
embedding layer, and the vector store. Its ``canonical_id`` feeds the
deterministic vector-store row ID; the git commit is deliberately NOT part of
the ID, so re-indexing a file at a new commit reuses stable row IDs and
stale chunks are removed via row filters (repository + file) before
the new ones are upserted. The commit stays in the payload for attribution.

Cross-referencing
-----------------
Every chunk carries ``symbols``: significant identifiers extracted from
its content (HDL construct names, function/class names in general code,
identifiers referenced in code fences inside documentation). The
retrieval service matches those lists across collections to attach
related HDL / documentation / code to search results.

HDL semantic model
------------------
All three HDL languages land in one collection. ``symbol_kind`` is a
*normalized* construct kind shared by the languages (``design_unit``,
``process``, ``package``, ``function``, ``task``, ...);
``native_symbol_kind`` keeps the language-specific name (``entity``,
``module``, ``always_ff``, ...) so a VHDL entity and an SV module are
both ``design_unit`` yet stay distinguishable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CollectionName(StrEnum):
    HDL = "hdl"  # VHDL + Verilog + SystemVerilog (one collection)
    DOCS = "docs"
    CODE = "code"


class ContentType(StrEnum):
    SOURCE = "source"  # HDL source (VHDL, Verilog, SystemVerilog)
    DOCUMENTATION = "documentation"  # markdown/text/RST
    CODE = "code"  # general source code (C/C++, Python, ...)


#: Index layout version (collection names + payload fields).
#:
#: v2 (current): one ``hdl`` collection holds VHDL, Verilog, and
#: SystemVerilog. v1 (legacy): a ``vhdl`` collection held VHDL only.
#: The index is derived from git, so upgrading past a layout change is a
#: safe deterministic full reindex (no manual data migration).
INDEX_SCHEMA_VERSION = 2

#: Maximum number of body lines quoted per search result.
#:
#: A chunk is a whole construct, and architectures/sections of 200-400
#: lines are routine, so an uncapped render made one default
#: (``limit=8``) call return 20-67 KB of text — several times the size
#: of ``grep -rn`` for the same identifier, which is why agents learned
#: to avoid the tool. Forty lines keeps the informative *head* of a
#: construct (entity header and port list, a process's sensitivity list
#: and first statements, a doc section's opening paragraphs) at roughly
#: 1.5-2 KB per hit, so a default call lands in the low thousands of
#: tokens instead of five figures; the elision marker names the exact
#: ``get_source`` call that returns the rest verbatim.
MAX_BODY_LINES = 40

#: Gutter width floor: line numbers are right-aligned in at least this
#: many columns so the quoted code stays vertically aligned.
_GUTTER_WIDTH = 4


def number_lines(text: str, start_line: int | None) -> str:
    """``text`` with an absolute, 1-based file line-number gutter.

    The rendered numbers are 1-based (the ``grep``/editor convention) —
    the LSP navigation tools (``find_definition``, ``find_references``,
    ``hover_info``) take **0-based** lines, so a line displayed as ``N``
    is passed to them as ``N - 1``. When ``start_line`` is unknown the
    text is returned unchanged: there is nothing truthful to number it
    with, and a wrong gutter is worse than none.
    """
    if start_line is None:
        return text
    lines = text.split("\n")
    width = max(len(str(start_line + len(lines) - 1)), _GUTTER_WIDTH)
    return "\n".join(
        f"{start_line + offset:>{width}} | {line}".rstrip()
        for offset, line in enumerate(lines)
    )


@dataclass(frozen=True)
class Chunk:
    """A semantically bounded, independently understandable text unit.

    HDL chunks: one per meaningful construct (VHDL entity/architecture/
    process, Verilog/SV module/interface/package/always block, ...) with
    parent context (library/entity/architecture/module, file, line range)
    so the chunk is understandable in isolation.
    Documentation chunks: one per heading/section/paragraph, with heading
    context and, when the section contains code snippets, the referenced
    identifiers in ``symbols``.
    Code chunks: one per function/class (Python, C/C++, ...), with the
    file/language in the payload and referenced identifiers in ``symbols``.
    """

    repository: str
    branch: str
    commit: str
    file: str
    content_type: ContentType
    language: str
    collection: CollectionName
    #: HDL: construct name. Docs: heading (or file stem). Code: function/
    #: class name (or file stem).
    symbol: str
    #: Normalized construct kind: design_unit, architecture, package,
    #: process, function, procedure, task, component, subprogram, file.
    #: Docs: section/paragraph. Code: function/class/file.
    symbol_kind: str
    start_line: int
    end_line: int
    content: str
    # HDL parent context
    library: str | None = None
    entity: str | None = None
    architecture: str | None = None
    #: Verilog/SV: the enclosing module for inner constructs.
    module: str | None = None
    #: Server-native construct kind (entity, module, always_ff, ...);
    #: distinguishes languages within a normalized kind.
    native_symbol_kind: str | None = None
    # Documentation context
    heading: str | None = None
    section: str | None = None
    #: Significant identifiers referenced/defined by this chunk
    #: (cross-referencing key, stored as a payload list).
    symbols: tuple[str, ...] = field(default_factory=tuple)

    @property
    def canonical_id(self) -> str:
        """Stable identity for deterministic point IDs (commit excluded).

        The line range disambiguates repeated names within one file (same
        section heading twice, same process label in two architectures of
        one file, two functions with the same name in a C file via
        different line ranges).
        """
        return "::".join(
            (
                self.repository,
                self.file,
                self.collection.value,
                self.symbol_kind,
                self.symbol,
                f"{self.start_line}-{self.end_line}",
            )
        )

    def payload(self) -> dict[str, Any]:
        """Full metadata + content stored as the vector-store row."""
        data: dict[str, Any] = {
            "repository": self.repository,
            "branch": self.branch,
            "commit": self.commit,
            "file": self.file,
            "content_type": self.content_type.value,
            "language": self.language,
            "collection": self.collection.value,
            "symbol": self.symbol,
            "symbol_kind": self.symbol_kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
        }
        for key in (
            "library",
            "entity",
            "architecture",
            "module",
            "native_symbol_kind",
            "heading",
            "section",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.symbols:
            data["symbols"] = list(self.symbols)
        return data


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval result with full source attribution."""

    result_type: str  # "hdl" | "docs" | "code"
    repository: str
    commit: str
    file: str
    content: str
    #: Relevance score. When the cross-encoder reranker ran (the
    #: default) it is that model's relevance in ``(0, 1)``, comparable
    #: across queries; otherwise it is the store's fused hybrid score
    #: (dense + full-text RRF, or cosine/BM25 for a single-leg mode),
    #: which is rank-derived and only comparable *within* one response.
    #: Either way it carries the bounded per-repository priority bonus.
    #: :attr:`SearchResults.calibrated_scores` says which of the two a
    #: given response holds.
    score: float
    language: str | None = None
    symbol: str | None = None
    symbol_kind: str | None = None
    native_symbol_kind: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    library: str | None = None
    entity: str | None = None
    architecture: str | None = None
    module: str | None = None
    heading: str | None = None
    section: str | None = None
    #: Identifiers this chunk references/defines (cross-references).
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def _elision_marker(self, elided: int) -> str:
        """The follow-up call that returns the lines this render dropped.

        Spelled as a ready-to-copy ``get_source`` call on this result's
        own repository/file/lines so the agent never has to reconstruct
        the arguments.
        """
        if self.start_line is None or self.end_line is None:
            call = f'get_source("{self.repository}", "{self.file}")'
        else:
            first = self.start_line + MAX_BODY_LINES
            last = max(self.end_line, first)
            call = f'get_source("{self.repository}", "{self.file}", {first}, {last})'
        return f"… {elided} more lines — {call} for the full text"

    def render(self) -> str:
        """Compact, attribution-complete rendering for MCP output.

        The quoted body carries a 1-based line-number gutter and is
        capped at :data:`MAX_BODY_LINES` lines; when lines are dropped
        the last body line is the exact ``get_source`` call that returns
        them.
        """
        where = f"{self.repository}:{self.file}"
        if self.start_line is not None and self.end_line is not None:
            where += f":{self.start_line}-{self.end_line}"
        title_parts: list[str] = [self.result_type]
        if self.entity:
            title_parts.append(f"entity {self.entity}")
        if self.architecture:
            title_parts.append(f"architecture {self.architecture}")
        if self.module:
            title_parts.append(f"module {self.module}")
        if self.symbol and self.symbol_kind:
            part = f"{self.symbol_kind} {self.symbol}"
            if self.native_symbol_kind and self.native_symbol_kind != self.symbol_kind:
                part += f" ({self.native_symbol_kind})"
            title_parts.append(part)
        elif self.symbol:
            title_parts.append(self.symbol)
        if self.heading:
            title_parts.append(self.heading)
        title = " / ".join(title_parts)
        refs = f"\n- references: {', '.join(self.symbols[:12])}" if self.symbols else ""
        content_lines = self.content.rstrip().split("\n")
        elided = len(content_lines) - MAX_BODY_LINES
        body = number_lines("\n".join(content_lines[:MAX_BODY_LINES]), self.start_line)
        if elided > 0:
            body += f"\n{self._elision_marker(elided)}"
        if self.result_type in ("hdl", "code"):
            fence = self.language if self.result_type == "hdl" else "code"
            body = f"```{fence}\n{body}\n```"
        return (
            f"## [{self.result_type}] {title}\n"
            f"- source: {where}\n"
            f"- repository: {self.repository} (commit {self.commit[:12]})\n"
            f"- score: {self.score:.4f}\n"
            f"{refs}\n\n{body}\n"
        )


class SearchResults(list[SearchResult]):
    """The ranked results of one search plus how to describe them.

    A plain ``list`` (every caller that only wants the hits keeps
    working) carrying the two facts the renderer cannot recover from the
    results themselves:

    ``has_more``
        Further candidates existed beyond the requested limit. The
        retrieval layer over-fetches by one and sets this only when it
        really saw more, so the "there is more" note is informative
        instead of appearing on almost every call.
    ``calibrated_scores``
        The scores are cross-encoder relevance in ``(0, 1)`` (the
        reranker actually ran), so an absolute weak-match threshold is
        meaningful. False when the ranking fell back to RRF/cosine/BM25,
        whose scale depends on the query and list lengths — callers must
        not compare those against a fixed threshold.
    """

    def __init__(
        self,
        results: Iterable[SearchResult] = (),
        *,
        has_more: bool = False,
        calibrated_scores: bool = False,
    ) -> None:
        super().__init__(results)
        self.has_more = has_more
        self.calibrated_scores = calibrated_scores
