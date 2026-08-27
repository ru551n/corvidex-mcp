"""Interpreter capability probes for the test suite.

Some CPython builds (e.g. some uv standalone 3.12/3.13 builds) link a
SQLite without loadable-extension support, so the stdlib
``sqlite3.Connection`` has no ``enable_load_extension`` and the
sqlite-vec extension cannot load. Store-dependent tests skip on those
interpreters instead of failing.
"""

from __future__ import annotations

import sqlite3


def sqlite_extensions_supported() -> bool:
    """True when this Python's stdlib SQLite can load loadable
    extensions (sqlite-vec); False otherwise."""
    conn = sqlite3.connect(":memory:")
    try:
        return hasattr(conn, "enable_load_extension")
    finally:
        conn.close()
