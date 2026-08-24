"""File-type routing: which collection a repository file belongs to.

Repositories are indexed for three domains — VHDL, VHDL-related
documentation, and general source code. A file's extension decides its
domain (and therefore its collection, content type, and language).
Per-repository ``domains`` select which of the three are loaded, and
``exclude`` patterns (fnmatch-style globs over the repository-relative
path) skip whole subtrees or file types. Files with no recognized
extension are not indexed.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
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


def is_excluded(path: str, patterns: Sequence[str]) -> bool:
    """True when any glob pattern matches the repository-relative path.

    Wildcard patterns are fnmatch-style and match across path separators,
    so ``"build/*"`` excludes everything under ``build/`` and
    ``"*.log"`` excludes any ``.log`` file at any depth. Patterns without
    wildcards match the path itself and its whole subtree (gitignore-style),
    so ``"build/sub"`` also excludes ``build/sub/fifo.vhd``.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        if not any(ch in pattern for ch in "*?[") and (
            path == pattern or path.startswith(f"{pattern}/")
        ):
            return True
    return False


def classify_file(
    path: str,
    domains: frozenset[CollectionName] | None = None,
    exclude: Sequence[str] = (),
) -> FileKind | None:
    """Map a repository-relative path to its domain, or None if not indexed.

    ``domains`` restricts the result to the repository's enabled
    collections (None = no restriction); ``exclude`` holds the repository's
    glob-style exclusion patterns.
    """
    if is_excluded(path, exclude):
        return None
    kind = _KIND_BY_EXTENSION.get(Path(path).suffix.lower())
    if kind is None:
        return None
    if domains is not None and kind.collection not in domains:
        return None
    return kind
