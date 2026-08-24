"""Shared typed models for VHDL, documentation, and code chunks.

The :class:`Chunk` model is the single currency between the indexers, the
embedding layer, and the vector store. Its ``canonical_id`` feeds the
deterministic Qdrant point ID; the git commit is deliberately NOT part of
the ID, so re-indexing a file at a new commit reuses stable point IDs and
stale chunks are removed via payload filters (repository + file) before
the new ones are upserted. The commit stays in the payload for attribution.

Cross-referencing
-----------------
Every chunk carries ``symbols``: significant identifiers extracted from
its content (VHDL construct names, function/class names in general code,
identifiers referenced in code fences inside documentation). The
retrieval service matches those lists across collections to attach
related VHDL / documentation / code to search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CollectionName(StrEnum):
    VHDL = "vhdl"
    DOCS = "docs"
    CODE = "code"


class ContentType(StrEnum):
    SOURCE = "source"  # VHDL
    DOCUMENTATION = "documentation"  # markdown/text/RST
    CODE = "code"  # general source code (C/C++, Python, ...)


@dataclass(frozen=True)
class SparseVectorData:
    """A sparse vector in storage-agnostic form (model vocab indices)."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def is_empty(self) -> bool:
        return not self.indices


@dataclass(frozen=True)
class Chunk:
    """A semantically bounded, independently understandable text unit.

    VHDL chunks: one per meaningful construct (entity, architecture,
    process, package, function, ...) with parent context (library/entity/
    architecture, file, line range) so the chunk is understandable in
    isolation.
    Documentation chunks: one per heading/section/paragraph, with heading
    context and, when the section contains code snippets, the referenced
    identifiers in ``symbols``.
    Code chunks: one per function/class (Python, C/C++, ...), with the
    file/language in the payload and referenced identifiers in ``symbols``.
    """

    repository: str
    repository_category: str
    repository_priority: int
    branch: str
    commit: str
    file: str
    content_type: ContentType
    language: str
    collection: CollectionName
    #: VHDL: construct name. Docs: heading (or file stem). Code: function/
    #: class name (or file stem).
    symbol: str
    #: VHDL: entity/architecture/process/package/...
    #: Docs: section/paragraph. Code: function/class/file.
    symbol_kind: str
    start_line: int
    end_line: int
    content: str
    # VHDL parent context
    library: str | None = None
    entity: str | None = None
    architecture: str | None = None
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
        """Full metadata + content stored as the Qdrant point payload."""
        data: dict[str, Any] = {
            "repository": self.repository,
            "repository_category": self.repository_category,
            "repository_priority": self.repository_priority,
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
        for key in ("library", "entity", "architecture", "heading", "section"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.symbols:
            data["symbols"] = list(self.symbols)
        return data


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval result with full source attribution."""

    result_type: str  # "vhdl" | "docs" | "code"
    repository: str
    repository_category: str
    commit: str
    file: str
    content: str
    #: Fused hybrid relevance score from the vector store (dense + sparse).
    store_score: float
    #: Final score after the bounded repository-priority adjustment.
    final_score: float
    symbol: str | None = None
    symbol_kind: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    library: str | None = None
    entity: str | None = None
    architecture: str | None = None
    heading: str | None = None
    section: str | None = None
    #: Identifiers this chunk references/defines (cross-references).
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Compact, attribution-complete rendering for MCP output."""
        where = f"{self.repository}:{self.file}"
        if self.start_line is not None and self.end_line is not None:
            where += f":{self.start_line}-{self.end_line}"
        title_parts: list[str] = [self.result_type]
        if self.entity:
            title_parts.append(f"entity {self.entity}")
        if self.architecture:
            title_parts.append(f"architecture {self.architecture}")
        if self.symbol and self.symbol_kind:
            title_parts.append(f"{self.symbol_kind} {self.symbol}")
        elif self.symbol:
            title_parts.append(self.symbol)
        if self.heading:
            title_parts.append(self.heading)
        title = " / ".join(title_parts)
        refs = f"\n- references: {', '.join(self.symbols[:12])}" if self.symbols else ""
        body = self.content.rstrip()
        if self.result_type in ("vhdl", "code"):
            body = f"```{self.result_type}\n{body}\n```"
        return (
            f"## [{self.result_type}] {title}\n"
            f"- source: {where}\n"
            f"- repository: {self.repository} ({self.repository_category}, "
            f"commit {self.commit[:12]})\n"
            f"- score: store={self.store_score:.4f} final={self.final_score:.4f}\n"
            f"{refs}\n\n{body}\n"
        )
