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

from vhdl_rag_mcp.config import AppConfig, RepositoryConfig
from vhdl_rag_mcp.embeddings.provider import FastEmbedProvider
from vhdl_rag_mcp.embeddings.providers import EmbeddingProviders
from vhdl_rag_mcp.git_manager import GitManager
from vhdl_rag_mcp.models import Chunk, CollectionName, ContentType
from vhdl_rag_mcp.retrieval import RetrievalError, RetrievalService
from vhdl_rag_mcp.state import StateStore
from vhdl_rag_mcp.vector_store import VectorStore

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


class FakeSparseVec:
    def __init__(self, indices, values) -> None:
        self.indices = np.asarray(indices, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)


class FakeSparse:
    def passage_embed(self, texts, mode="passage"):
        for text in texts:
            yield FakeSparseVec([len(text), len(text) + 7], [1.0, 2.0])

    def query_embed(self, query, mode="query"):
        yield FakeSparseVec([len(query)], [1.0])


def fake_providers(config: AppConfig) -> EmbeddingProviders:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider(
        "fake/dense", "fake/sparse", dense=FakeDense(), sparse=FakeSparse()
    )
    sparse = FastEmbedProvider("fake/sparse", "fake/sparse", sparse=FakeSparse())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    providers._sparse_provider = lambda: sparse  # type: ignore[method-assign]
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
) -> Chunk:
    content_type = {
        CollectionName.VHDL: ContentType.SOURCE,
        CollectionName.DOCS: ContentType.DOCUMENTATION,
        CollectionName.CODE: ContentType.CODE,
    }[collection]
    language = {
        CollectionName.VHDL: "vhdl",
        CollectionName.DOCS: "markdown",
        CollectionName.CODE: "c",
    }[collection]
    return Chunk(
        repository="repo",
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
    store.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    providers = fake_providers(config)
    git_manager = GitManager(config.repos_dir)
    states = StateStore(config.state_dir / "repositories.json")

    cfg = config.repository("repo")
    plan = await git_manager.sync(cfg, None)
    states.set_indexed("repo", plan.commit)

    chunks = [
        make_chunk(
            CollectionName.VHDL,
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
            CollectionName.VHDL,
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
    dense = providers.embed_passages(CollectionName.VHDL, [c.content for c in chunks])
    sparse = providers.embed_sparse_passages([c.content for c in chunks])
    store.upsert_chunks(chunks, dense, sparse)

    retrieval = RetrievalService(config, git_manager, store, providers, states)
    yield store, retrieval
    store.close()


# -- tests ---------------------------------------------------------------------


async def test_search_returns_ranked_results(env) -> None:
    _store, retrieval = env
    results = retrieval.search(CollectionName.VHDL, "fifo entity")
    assert results
    top = results[0]
    assert top.result_type == "vhdl"
    assert top.repository == "repo"
    assert top.commit
    assert top.content
    assert top.score > 0
    assert all(r.result_type == "vhdl" for r in results)


async def test_search_symbol_cross_reference(env) -> None:
    _store, retrieval = env
    results = retrieval.search(CollectionName.VHDL, "reset", symbols=("rst_n",))
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
        retrieval.search(CollectionName.VHDL, "x", repository="nope")
    with pytest.raises(RetrievalError, match="must not be empty"):
        retrieval.search(CollectionName.VHDL, "   ")
    filtered = retrieval.search(CollectionName.CODE, "fifo", repository="repo")
    assert filtered
    assert all(r.repository == "repo" for r in filtered)


async def test_search_knowledge_fuses_domains(env) -> None:
    _store, retrieval = env
    results = retrieval.search_knowledge("fifo", limit=20)
    assert {r.result_type for r in results} == {"vhdl", "docs", "code"}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_search_knowledge_respects_limit(env) -> None:
    _store, retrieval = env
    assert len(retrieval.search_knowledge("fifo", limit=2)) <= 2


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


async def test_repository_status(env) -> None:
    _store, retrieval = env
    statuses = retrieval.repository_status()
    assert len(statuses) == 1
    st = statuses[0]
    assert st.name == "repo"
    assert st.ref == "main"
    assert st.domains == ("vhdl", "docs", "code")
    assert st.indexed_commit is not None
    assert st.last_sync_error is None
