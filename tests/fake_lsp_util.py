"""Shared helper: a fake LSP server script executable on the test platform."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


def executable_lsp_script(tmp_path: Path, name: str, source: str) -> Path:
    """Write a fake LSP script and return a path the platform can execute.

    POSIX: the script itself (shebang + execute bit). Windows has no
    shebang execution, so a ``.cmd`` wrapper that runs the script with
    the current interpreter and forwards all arguments is returned.
    """
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / f"{name}.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
        return wrapper
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script
