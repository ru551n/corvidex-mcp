"""Tests for the retrieval service (hybrid search, fusion, source access).

Runs fully offline: a local file:// git remote and fake embedding
providers. Chunks are upserted directly (the pipeline is covered in
test_pipeline.py), so this file exercises the retrieval layer itself.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from capability import sqlite_extensions_supported

from corvidex_mcp.config import AppConfig, RepositoryConfig
from corvidex_mcp.embeddings.provider import FastEmbedProvider
from corvidex_mcp.embeddings.providers import EmbeddingProviders
from corvidex_mcp.git_manager import GitManager
from corvidex_mcp.models import Chunk, CollectionName, ContentType
from corvidex_mcp.retrieval import RetrievalError, RetrievalService
from corvidex_mcp.state import StateStore
from corvidex_mcp.vector_store import VectorStore

pytestmark = pytest.mark.skipif(
    not sqlite_extensions_supported(),
    reason=(
        "stdlib SQLite lacks loadable-extension support (the sqlite-vec "
        "extension cannot load; use CPython 3.14 or a system/homebrew "
        "Python)"
    ),
)


ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}

ENTITY_CONTENT = "entity fifo is\n  port (clk : in std_logic);\nend entity fifo;\n"
ARCH_CONTENT = "architecture rtl of fifo is\nbegin\nend architecture rtl;\n"
FIFO_VHDL = ENTITY_CONTENT + "\n" + ARCH_CONTENT
STD_MD = "# Standard\n\n## Reset conventions\n\nAsync resets are named rst_n.\n"
FIFO_C = "int fifo_write(int *mem) {\n    return 0;\n}\n"
FIFO_TB_V = (
    "module fifo_tb (\n  input logic clk,\n"
    "  output logic [FIFO_DEPTH-1:0] dout\n);\nendmodule"
)
FIFO_PKG_SV = "package fifo_pkg;\n  localparam int FIFO_DEPTH = 8;\nendpackage"


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# -- fake embeddings -------------------------------------------------------


class FakeDense:
    embedding_size = 4

    def passage_embed(self, texts, batch_size=32):
        for i, text in enumerate(texts):
            yield np.array([float(len(text)), float(i), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, query, batch_size=32):
        yield np.array([float(len(query)), 0.0, 0.0, 0.0], dtype=np.float32)


def fake_providers(config: AppConfig) -> EmbeddingProviders:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    return providers


def make_chunk(
    collection: CollectionName,
    file: str,
    symbol: str,
    kind: str,
    start: int,
    end: int,
    content: str,
    commit: str,
    symbols: tuple[str, ...] = (),
    language: str | None = None,
    repository: str = "repo",
) -> Chunk:
    content_type = {
        CollectionName.HDL: ContentType.SOURCE,
        CollectionName.DOCS: ContentType.DOCUMENTATION,
        CollectionName.CODE: ContentType.CODE,
    }[collection]
    if language is None:
        language = {
            CollectionName.HDL: "vhdl",
            CollectionName.DOCS: "markdown",
            CollectionName.CODE: "c",
        }[collection]
    return Chunk(
        repository=repository,
        branch="main",
        commit=commit,
        file=file,
        content_type=content_type,
        language=language,
        collection=collection,
        symbol=symbol,
        symbol_kind=kind,
        start_line=start,
        end_line=end,
        content=content,
        heading=kind if collection is CollectionName.DOCS else None,
        symbols=symbols,
    )


@pytest.fixture
async def env(tmp_path: Path):
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.vhd").write_text(FIFO_VHDL)
    (up / "docs").mkdir()
    (up / "docs" / "standard.md").write_text(STD_MD)
    (up / "src").mkdir()
    (up / "src" / "fifo.c").write_text(FIFO_C)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")

    config = AppConfig(
        data_dir=tmp_path / "data",
        repositories=[RepositoryConfig(name="repo", url=str(up), ref="main")],
    )
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    providers = fake_providers(config)
    git_manager = GitManager(config.repos_dir)
    states = StateStore(config.sqlite_index_path)

    cfg = config.repository("repo")
    plan = await git_manager.sync(cfg, None)
    states.set_indexed("repo", plan.commit)

    chunks = [
        make_chunk(
            CollectionName.HDL,
            "rtl/fifo.vhd",
            "fifo",
            "entity",
            1,
            3,
            ENTITY_CONTENT,
            plan.commit,
            symbols=("fifo", "clk", "std_logic"),
        ),
        make_chunk(
            CollectionName.HDL,
            "rtl/fifo.vhd",
            "rtl",
            "architecture",
            5,
            7,
            ARCH_CONTENT,
            plan.commit,
            symbols=("fifo", "rtl"),
        ),
        make_chunk(
            CollectionName.DOCS,
            "docs/standard.md",
            "Standard",
            "section",
            1,
            5,
            STD_MD,
            plan.commit,
            symbols=("rst_n",),
        ),
        make_chunk(
            CollectionName.HDL,
            "tb/fifo_tb.v",
            "fifo_tb",
            "design_unit",
            1,
            6,
            FIFO_TB_V,
            plan.commit,
            symbols=("fifo_tb", "FIFO_DEPTH", "clk"),
            language="verilog",
        ),
        make_chunk(
            CollectionName.HDL,
            "rtl/fifo_pkg.sv",
            "fifo_pkg",
            "package",
            1,
            4,
            FIFO_PKG_SV,
            plan.commit,
            symbols=("fifo_pkg", "FIFO_DEPTH"),
            language="systemverilog",
        ),
        make_chunk(
            CollectionName.CODE,
            "src/fifo.c",
            "fifo_write",
            "function",
            1,
            3,
            FIFO_C,
            plan.commit,
            symbols=("fifo_write", "mem"),
        ),
    ]
    dense = providers.embed_passages(CollectionName.HDL, [c.content for c in chunks])
    store.upsert_chunks(chunks, dense)

    retrieval = RetrievalService(config, git_manager, store, providers, states)
    yield store, retrieval
    store.close()


# -- tests ---------------------------------------------------------------------


async def test_search_returns_ranked_results(env) -> None:
    _store, retrieval = env
    results = retrieval.search(CollectionName.HDL, "fifo entity")
    assert results
    top = results[0]
    assert top.result_type == "hdl"
    assert top.repository == "repo"
    assert top.commit
    assert top.content
    assert top.score > 0
    assert all(r.result_type == "hdl" for r in results)


async def test_search_symbol_cross_reference(env) -> None:
    _store, retrieval = env
    results = retrieval.search(CollectionName.HDL, "reset", symbols=("rst_n",))
    assert not results  # no VHDL chunk references rst_n
    docs = retrieval.search(
        CollectionName.DOCS, "reset conventions", symbols=("rst_n",)
    )
    assert docs
    assert docs[0].file == "docs/standard.md"
    assert "rst_n" in docs[0].symbols


async def test_search_repository_filter_and_errors(env) -> None:
    _store, retrieval = env
    with pytest.raises(RetrievalError, match="unknown repository"):
        retrieval.search(CollectionName.HDL, "x", repository="nope")
    with pytest.raises(RetrievalError, match="must not be empty"):
        retrieval.search(CollectionName.HDL, "   ")
    filtered = retrieval.search(CollectionName.CODE, "fifo", repository="repo")
    assert filtered
    assert all(r.repository == "repo" for r in filtered)


async def test_search_knowledge_fuses_domains(env) -> None:
    _store, retrieval = env
    results = retrieval.search_knowledge("fifo", limit=20)
    assert {r.result_type for r in results} == {"hdl", "docs", "code"}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_search_knowledge_respects_limit(env) -> None:
    _store, retrieval = env
    assert len(retrieval.search_knowledge("fifo", limit=2)) <= 2


async def test_search_language_filter_per_language(env) -> None:
    _store, retrieval = env
    for language in ("vhdl", "verilog", "systemverilog"):
        results = retrieval.search(CollectionName.HDL, "fifo", language=language)
        assert results, language
        assert all(r.language == language for r in results), language


async def test_search_language_filter_is_exact(env) -> None:
    _store, retrieval = env
    # 'verilog' must not match the systemverilog or vhdl chunks.
    results = retrieval.search(CollectionName.HDL, "fifo", language="verilog")
    assert all(r.language == "verilog" for r in results)
    files = {r.file for r in results}
    assert "rtl/fifo_pkg.sv" not in files
    assert "rtl/fifo.vhd" not in files


async def test_search_language_validation(env) -> None:
    _store, retrieval = env
    with pytest.raises(RetrievalError, match="unknown HDL language"):
        retrieval.search(CollectionName.HDL, "fifo", language="verilog-2005")
    with pytest.raises(RetrievalError, match="must not be empty"):
        retrieval.search(CollectionName.HDL, "fifo", language="  ")
    # Non-hdl collections accept pass-through languages.
    assert retrieval.search(CollectionName.CODE, "fifo", language="c")


async def test_search_knowledge_language_filter(env) -> None:
    _store, retrieval = env
    results = retrieval.search_knowledge("fifo", limit=20, language="c")
    assert results
    assert all(r.language == "c" for r in results)
    # The hdl domain simply contributes nothing for a code language.
    assert all(r.result_type != "hdl" for r in results)


async def test_results_carry_language_metadata(env) -> None:
    _store, retrieval = env
    results = retrieval.search(CollectionName.HDL, "fifo", limit=10)
    assert results
    assert {r.language for r in results} <= {"vhdl", "verilog", "systemverilog"}
    # Every hdl result has a language; cross-domain too.
    knowledge = retrieval.search_knowledge("fifo", limit=20)
    assert all(r.language for r in knowledge)


async def test_get_source(env) -> None:
    _store, retrieval = env
    text = retrieval.get_source("repo", "rtl/fifo.vhd")
    assert text.startswith("repo:rtl/fifo.vhd @ ")
    assert FIFO_VHDL.rstrip() in text
    sliced = retrieval.get_source("repo", "rtl/fifo.vhd", start_line=2, end_line=3)
    assert "(lines 2-3" in sliced
    assert sliced.endswith("end entity fifo;")
    with pytest.raises(RetrievalError, match="not found"):
        retrieval.get_source("repo", "no/such.c")
    with pytest.raises(RetrievalError, match="unknown repository"):
        retrieval.get_source("nope", "rtl/fifo.vhd")


async def test_get_source_bad_range(env) -> None:
    _store, retrieval = env
    with pytest.raises(RetrievalError, match="invalid line range"):
        retrieval.get_source("repo", "rtl/fifo.vhd", start_line=50, end_line=60)


async def test_list_files(env) -> None:
    """The indexed files are listable (the candidate paths for get_source),
    glob-filtered and capped."""
    _store, retrieval = env
    files, truncated = retrieval.list_files("repo")
    assert files == [
        "docs/standard.md",
        "rtl/fifo.vhd",
        "rtl/fifo_pkg.sv",
        "src/fifo.c",
        "tb/fifo_tb.v",
    ]
    assert truncated is False
    files, _ = retrieval.list_files("repo", pattern="*.vhd")
    assert files == ["rtl/fifo.vhd"]
    files, _ = retrieval.list_files("repo", pattern="rtl/*")
    assert files == ["rtl/fifo.vhd", "rtl/fifo_pkg.sv"]
    files, _ = retrieval.list_files("repo", pattern="nope/*")
    assert files == []
    files, truncated = retrieval.list_files("repo", limit=2)
    assert files == ["docs/standard.md", "rtl/fifo.vhd"]
    assert truncated is True
    with pytest.raises(RetrievalError, match="unknown repository"):
        retrieval.list_files("nope")


async def test_repository_status(env) -> None:
    _store, retrieval = env
    statuses = retrieval.repository_status()
    assert len(statuses) == 1
    st = statuses[0]
    assert st.name == "repo"
    assert st.ref == "main"
    assert st.domains == ("hdl", "docs", "code")
    assert st.priority == 1
    assert st.indexed_commit is not None
    assert st.last_sync_error is None


# -- repository priority (bounded post-RRF bonus) ---------------------------


async def _priority_env(
    tmp_path: Path, priorities: dict[str, int], contents: dict[str, str]
) -> tuple[VectorStore, RetrievalService, StateStore, AppConfig]:
    """One git remote; each repository name indexes its own copy of the
    content under rtl/<name>.vhd (identical remote for all repos)."""
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    for name, content in contents.items():
        (up / "rtl" / f"{name}.vhd").write_text(content)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")

    config = AppConfig(
        data_dir=tmp_path / "data",
        repositories=[
            RepositoryConfig(
                name=name,
                url=str(up),
                ref="main",
                priority=priorities.get(name, 1),
            )
            for name in priorities
        ],
    )
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    providers = fake_providers(config)
    git_manager = GitManager(config.repos_dir)
    states = StateStore(config.sqlite_index_path)

    chunks = []
    for name in priorities:
        plan = await git_manager.sync(config.repository(name), None)
        states.set_indexed(name, plan.commit)
        chunks.append(
            make_chunk(
                CollectionName.HDL,
                f"rtl/{name}.vhd",
                name,
                "design_unit",
                1,
                1,
                contents[name],
                plan.commit,
                repository=name,
            )
        )
    dense = providers.embed_passages(CollectionName.HDL, [c.content for c in chunks])
    store.upsert_chunks(chunks, dense)
    return (
        store,
        RetrievalService(config, git_manager, store, providers, states),
        states,
        config,
    )


async def test_priority_bonus_adds_one_step_per_unit(tmp_path: Path) -> None:
    # Neither chunk matches the query's terms: the full-text leg is empty
    # for both, so the fused RRF score is the single dense-leg rank term.
    # "base" (priority 1) is the denser vector (rank 1, 1/61); "extra"
    # (priority 2) is rank 2 (1/62) plus exactly one bonus step — which
    # overtakes the one-rank relevance gap.
    store, retrieval, _states, _config = await _priority_env(
        tmp_path,
        priorities={"base": 1, "extra": 2},
        contents={"base": "gamma delta", "extra": "gamma delta extra"},
    )
    step = 1.0 / (60 * 61)
    results = retrieval.search(CollectionName.HDL, "alpha beta")
    assert [r.repository for r in results] == ["extra", "base"]
    assert results[0].score == pytest.approx(1.0 / 62 + step)
    assert results[1].score == pytest.approx(1.0 / 61)
    # The same bonus applies in cross-domain fusion.
    fused = retrieval.search_knowledge("alpha beta")
    assert [r.repository for r in fused] == ["extra", "base"]
    store.close()


async def test_priority_bonus_saturates_and_cannot_cross_tiers(
    tmp_path: Path,
) -> None:
    # "anchor" matches the query (rank 1 in both legs: 2/61). "extra" has
    # an extreme priority (1000): its bonus saturates at the cap, which
    # is far too small to promote its single-list score past the
    # two-list anchor — the boundedness guarantee.
    store, retrieval, _states, _config = await _priority_env(
        tmp_path,
        priorities={"anchor": 1, "base": 1, "extra": 1000},
        contents={
            "anchor": "alpha beta",
            "base": "gamma delta",
            "extra": "gamma delta extra",
        },
    )
    cap = 0.25 / 61
    results = retrieval.search(CollectionName.HDL, "alpha beta")
    assert results[0].repository == "anchor"
    by_repo = {r.repository: r.score for r in results}
    # Dense ranks: anchor (len 10, index 0), base (len 10, index 1),
    # extra (len 16, index 2); FTS leg: anchor only.
    assert by_repo["extra"] == pytest.approx(1.0 / 63 + cap)
    assert by_repo["anchor"] == pytest.approx(2.0 / 61)
    assert by_repo["anchor"] > by_repo["extra"]
    # Saturation: 999 steps would be ~0.27; the bonus is capped.
    assert cap < 999 * (1.0 / (60 * 61))
    store.close()


# -- search modes (semantic / lexical / hybrid) -----------------------------


async def test_semantic_mode_is_dense_only(env) -> None:
    _store, retrieval = env
    # A query with no full-text matches: the FTS leg is empty anyway, so
    # the ordering is pure embedding similarity. The entity chunk's
    # vector is the query-aligned one (first index, longest content):
    # score 1.0, the rest within [0, 1] below it.
    results = retrieval.search(CollectionName.HDL, "zzzqqq", mode="semantic")
    assert results
    assert results[0].file == "rtl/fifo.vhd"
    assert results[0].start_line == 1
    assert results[0].score == pytest.approx(1.0)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.score < results[0].score for r in results[1:])
    # Case-insensitive mode names.
    again = retrieval.search(CollectionName.HDL, "zzzqqq", mode="SEMANTIC")
    assert [r.file for r in again] == [r.file for r in results]


async def test_lexical_mode_is_fulltext_only(env) -> None:
    _store, retrieval = env
    # "entity OR fifo": only the two fifo.vhd design units contain those
    # tokens (fifo_tb / fifo_pkg are single '_'-joined tokens). Ranked
    # by BM25; the display score is the rank's RRF term.
    results = retrieval.search(CollectionName.HDL, "entity fifo", mode="lexical")
    assert [(r.file, r.start_line) for r in results] == [
        ("rtl/fifo.vhd", 1),
        ("rtl/fifo.vhd", 5),
    ]
    assert results[0].score == pytest.approx(1.0 / 61)
    assert results[1].score == pytest.approx(1.0 / 62)
    # No full-text matches: the lexical leg is empty (no dense fallback).
    assert retrieval.search(CollectionName.HDL, "zzzqqq", mode="lexical") == []


async def test_hybrid_mode_is_the_default(env) -> None:
    _store, retrieval = env
    default = retrieval.search(CollectionName.HDL, "entity fifo")
    explicit = retrieval.search(CollectionName.HDL, "entity fifo", mode="hybrid")
    assert [(r.file, r.start_line) for r in default] == [
        (r.file, r.start_line) for r in explicit
    ]
    # Both legs contribute: the top score is the sum of two RRF terms.
    assert default[0].score == pytest.approx(2.0 / 61)
    # Hybrid is distinct from lexical on a query where the dense leg
    # disagrees (the entity chunk is dense-best here too, but the
    # scores live on different scales).
    lexical = retrieval.search(CollectionName.HDL, "entity fifo", mode="lexical")
    assert default[0].score != lexical[0].score


async def test_invalid_mode_is_rejected(env) -> None:
    _store, retrieval = env
    with pytest.raises(RetrievalError, match="unknown search mode"):
        retrieval.search(CollectionName.HDL, "fifo", mode="cosine")
    with pytest.raises(RetrievalError, match="unknown search mode"):
        retrieval.search_knowledge("fifo", mode="")


async def test_search_knowledge_modes(env) -> None:
    _store, retrieval = env
    # Lexical across domains: only hdl chunks contain the tokens.
    lexical = retrieval.search_knowledge("entity fifo", mode="lexical")
    assert [r.result_type for r in lexical] == ["hdl", "hdl"]
    assert [(r.file, r.start_line) for r in lexical] == [
        ("rtl/fifo.vhd", 1),
        ("rtl/fifo.vhd", 5),
    ]
    # Semantic across domains: every collection contributes its dense
    # top; scores are cosine similarities in [0, 1].
    semantic = retrieval.search_knowledge("zzzqqq", mode="semantic")
    assert semantic
    assert {r.result_type for r in semantic} == {"hdl", "docs", "code"}
    assert all(0.0 <= r.score <= 1.0 for r in semantic)
