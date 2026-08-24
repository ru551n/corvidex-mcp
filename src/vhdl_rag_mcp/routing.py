"""File-type routing: which collection a repository file belongs to.

Repositories are indexed for three domains — VHDL, VHDL-related
documentation, and general source code. A file's extension decides its
domain (and therefore its collection, content type, and language); files
with no recognized extension are not indexed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import CollectionName, ContentType

VHDL_EXTENSIONS: frozenset[str] = frozenset({".vhd", ".vhdl"})
DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".rst", ".txt"})
CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cuh", ".py"}
)

_DOC_LANGUAGES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
}
_CODE_LANGUAGES = {
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cuh": "cpp",
    ".py": "python",
}


@dataclass(frozen=True)
class FileKind:
    """The indexing domain of a file."""

    content_type: ContentType
    collection: CollectionName
    language: str


def _build_table() -> dict[str, FileKind]:
    table: dict[str, FileKind] = {}
    for ext in VHDL_EXTENSIONS:
        table[ext] = FileKind(ContentType.SOURCE, CollectionName.VHDL, "vhdl")
    for ext in DOC_EXTENSIONS:
        table[ext] = FileKind(
            ContentType.DOCUMENTATION, CollectionName.DOCS, _DOC_LANGUAGES[ext]
        )
    for ext in CODE_EXTENSIONS:
        table[ext] = FileKind(
            ContentType.CODE, CollectionName.CODE, _CODE_LANGUAGES[ext]
        )
    return table


_KIND_BY_EXTENSION: dict[str, FileKind] = _build_table()


def classify_file(path: str) -> FileKind | None:
    """Map a repository-relative path to its domain, or None if not indexed."""
    return _KIND_BY_EXTENSION.get(Path(path).suffix.lower())
