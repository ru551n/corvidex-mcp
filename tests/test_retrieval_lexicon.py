"""Tests for the static RTL/HDL query-expansion lexicon."""

from __future__ import annotations

from corvidex_mcp.retrieval_lexicon import MAX_EXTRA_TERMS, expand_query


def test_expand_appends_matched_synonyms() -> None:
    expanded = expand_query("asynchronous reset")
    assert expanded.startswith("asynchronous reset ")
    assert "rst" in expanded.split()
    assert "async" in expanded.split()


def test_expand_is_case_insensitive() -> None:
    expanded = expand_query("Clock domain")
    assert "clk" in expanded.split()


def test_expand_no_match_returns_query_unchanged() -> None:
    assert expand_query("fibonacci sequence") == "fibonacci sequence"


def test_expand_skips_exact_identifier_queries() -> None:
    assert expand_query("rst_n") == "rst_n"
    assert expand_query("fifo_write") == "fifo_write"


def test_expand_empty_query_unchanged() -> None:
    assert expand_query("") == ""
    assert expand_query("   ") == "   "


def test_expand_does_not_duplicate_terms_already_present() -> None:
    expanded = expand_query("reset and rst_n")
    # "rst_n" is already present verbatim: not appended again.
    assert expanded.count("rst_n") == 1


def test_expand_never_removes_original_terms() -> None:
    query = "generic parameter width"
    expanded = expand_query(query)
    assert expanded.startswith(query)


def test_expand_caps_extra_terms() -> None:
    # A query that touches many lexicon entries at once.
    query = (
        "reset clock asynchronous synchronous handshake flip-flop "
        "generic testbench fifo state machine signal"
    )
    expanded = expand_query(query)
    extra = expanded[len(query) :].split()
    assert len(extra) <= MAX_EXTRA_TERMS
