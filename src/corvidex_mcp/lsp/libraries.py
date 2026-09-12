"""VHDL library inference for the generated ``vhdl_ls.toml``.

vhdl_ls resolves a *library-qualified* name (``cnn_accel.cnn_accel_pkg``,
``common.attribute_pkg``, ``fifo.asynchronous_fifo``) by looking up the
library in its ``vhdl_ls.toml``. A configuration that declares every
file in one ``defaultlib`` therefore cannot resolve any of them: only
``work.<name>`` (``work`` is always the library the current file belongs
to) and unqualified names work. Library-qualified entity instantiation is
the standard tsfpga/hdl-modules style, so a single-library configuration
silently breaks go-to-definition, find-references and hover for the
dominant case in those projects.

This module infers the library each file belongs to from the directory
layout so the generated configuration can declare one
``[libraries.<name>]`` section per library instead.

The layout is tsfpga's (``tsfpga/module.py``,
``tsfpga/create_vhdl_ls_config.py``): a ``modules`` directory holds one
sub-directory per library, the sub-directory's name *is* the library
name, and everything below it — ``src/``, ``rtl/``, ``test/``, ``sim/``,
``regs_src/``, ``regs_sim/``, ``hdl/`` … — belongs to that one library.
tsfpga's own generator emits a single ``<module>/**/*.vhd`` glob per
module for exactly that reason. Both repositories in a typical checkout
follow it, including a vendored one under a submodule prefix
(``hdl-modules/modules/fifo/src/asynchronous_fifo.vhd`` is library
``fifo``), which is why the ``modules`` segment is matched anywhere in
the path rather than only at the root.

Anything that does not match falls back to :data:`DEFAULT_LIBRARY`,
preserving the previous single-library behaviour for layouts this does
not recognise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath

#: Library that holds every file whose path matches no recognised
#: layout (and the whole workspace when inference is not used at all).
DEFAULT_LIBRARY = "defaultlib"

#: Directory holding one sub-directory per library (tsfpga convention).
MODULES_DIR = "modules"

#: A VHDL basic identifier: letter, then letters/digits/underscores.
#: Library names that are not one of these (``__pycache__``, ``.cache``,
#: ``2020-archive``) are not plausible library directories.
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: Names a generated library section must never take. ``work`` is
#: rejected outright by vhdl_ls ("The 'work' library is not a valid
#: library") and a single bad name invalidates the *entire*
#: configuration; ``std``/``ieee`` are emitted separately for the
#: bundled ``vhdl_libraries`` and must not be redefined. Files under a
#: directory with one of these names fall back to
#: :data:`DEFAULT_LIBRARY`.
RESERVED_LIBRARY_NAMES = frozenset({"work", "std", "ieee"})


def _is_library_name(name: str) -> bool:
    return (
        _IDENTIFIER.fullmatch(name) is not None
        and name.lower() not in RESERVED_LIBRARY_NAMES
    )


def infer_library(relative_path: str) -> str | None:
    """The VHDL library ``relative_path`` belongs to, or None.

    Returns the name of the directory immediately below the innermost
    ``modules/`` segment that still has the file somewhere beneath it —
    innermost so a vendored checkout's own ``modules/`` wins over the
    outer repository's (``hdl-modules/modules/fifo/src/x.vhd`` is
    ``fifo``, not ``hdl-modules``). None when the path matches no
    recognised layout, or when the candidate name could not be used as
    a library name (see :data:`RESERVED_LIBRARY_NAMES`).
    """
    parts = PurePosixPath(relative_path.replace("\\", "/")).parts
    # parts[index] == "modules", parts[index + 1] is the library, and at
    # least one more segment (the file itself, possibly under src/…).
    for index in range(len(parts) - 3, -1, -1):
        if parts[index] != MODULES_DIR:
            continue
        candidate = parts[index + 1]
        if _is_library_name(candidate):
            return candidate
    return None


def group_by_library(files: Iterable[str]) -> dict[str, list[str]]:
    """Map library name -> the given files that belong to it.

    Files matching no recognised layout are collected under
    :data:`DEFAULT_LIBRARY`. Ordering is deterministic:
    :data:`DEFAULT_LIBRARY` first (when non-empty), then the inferred
    libraries alphabetically, each with its files in the order given.
    """
    groups: dict[str, list[str]] = {}
    for file in files:
        groups.setdefault(infer_library(file) or DEFAULT_LIBRARY, []).append(file)
    ordered: dict[str, list[str]] = {}
    if DEFAULT_LIBRARY in groups:
        ordered[DEFAULT_LIBRARY] = groups.pop(DEFAULT_LIBRARY)
    for name in sorted(groups):
        ordered[name] = groups[name]
    return ordered


def toml_string(value: str) -> str:
    """``value`` as a TOML string.

    A literal string (single quotes) keeps Windows path separators
    intact, so it is preferred; a value a literal string cannot express
    falls back to an escaped basic string.
    """
    if "'" not in value and "\n" not in value and "\r" not in value:
        return f"'{value}'"
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def library_section(
    name: str, files: Iterable[str], *, third_party: bool = False
) -> list[str]:
    """The ``[libraries.<name>]`` lines for one library (blank-terminated)."""
    entries = ", ".join(toml_string(file) for file in files)
    lines = [f"[libraries.{name}]", f"files = [{entries}]"]
    if third_party:
        lines.append("is_third_party = true")
    lines.append("")
    return lines
