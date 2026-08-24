"""Logging setup for the MCP stdio server.

stdout is reserved exclusively for MCP protocol traffic; anything printed
there corrupts the protocol. All logging goes to stderr and, optionally,
to a rotating log file.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Third-party loggers kept at WARNING to keep stderr readable.
_NOISY_LOGGERS = (
    "qdrant_client",
    "fastembed",
    "onnxruntime",
    "huggingface_hub",
    "httpx",
    "urllib3",
)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging to stderr (and optionally a rotating file).

    Idempotent: existing handlers are replaced, so repeated calls (e.g. in
    tests) do not duplicate output.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
