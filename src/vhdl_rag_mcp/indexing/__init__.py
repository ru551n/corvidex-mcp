"""Indexing pipeline: per-domain chunkers and the incremental sync driver."""

from .vhdl import chunk_vhdl_file, extract_identifiers

__all__ = ["chunk_vhdl_file", "extract_identifiers"]
