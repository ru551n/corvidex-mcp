"""Smoke tests for the package skeleton."""

from __future__ import annotations

import logging

from corvidex_mcp import __version__
from corvidex_mcp.logging_setup import setup_logging


def test_version() -> None:
    assert __version__


def test_logging_goes_to_stderr_only(capfd) -> None:
    setup_logging("DEBUG")
    logging.getLogger("corvidex_mcp.test").info("hello stderr")
    captured = capfd.readouterr()
    assert "hello stderr" in captured.err
    assert captured.out == ""


def test_logging_file(tmp_path) -> None:
    log_file = tmp_path / "logs" / "test.log"
    setup_logging("INFO", log_file=log_file)
    logging.getLogger("corvidex_mcp.test").info("in file")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file.exists()
    assert "in file" in log_file.read_text()


def test_logging_idempotent(capfd) -> None:
    setup_logging("DEBUG")
    setup_logging("DEBUG")
    logging.getLogger("corvidex_mcp.test").info("once only")
    captured = capfd.readouterr()
    assert captured.err.count("once only") == 1
