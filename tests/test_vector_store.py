"""Tests for the Qdrant-backed vector store (local mode, temp directories)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vhdl_rag_mcp.config import AppConfig
from vhdl_rag_mcp.models import (
    Chunk,
    CollectionName,
    ContentType,
    SearchResult,
    SparseVectorData,
)
from vhdl_rag_mcp.vector_store import VectorStore, VectorStoreError, point_id


def make_chunk(
    repo: str = "repoA",
    file: str = "rtl/fifo.vhd",
    symbol: str = "p_write",
    kind: str = "process",
    start: int = 10,
    end: int = 20,
    content: str = "process body",
    collection: CollectionName = CollectionName.VHDL,
    content_type: ContentType = ContentType.SOURCE,
    language: str = "vhdl",
    symbols: tuple[str, ...] = (),
    **extra: object,
) -> Chunk:
    return Chunk(
        repository=repo,
        repository_category="approved",
        repository_priority=90,
        branch="main",
        commit="abc123",
        file=file,
        content_type=content_type,
        language=language,
        collection=collection,
        symbol=symbol,
        symbol_kind=kind,
        start_line=start,
        end_line=end,
        content=content,
        library=extra.get("library"),
        entity=extra.get("entity"),
        architecture=extra.get("architecture"),
        heading=extra.get("heading"),
        section=extra.get("section"),
        symbols=symbols,
    )


def dense(i: int) -> list[float]:
    return [float(i), 0.0, 0.0, 0.0]


def sparse(
    indices: tuple[int, ...] = (1,),
    values: tuple[float, ...] | None = None,
) -> SparseVectorData:
    if values is None:
        values = tuple(1.0 for _ in indices)
    return SparseVectorData(indices=indices, values=values)


@pytest.fixture
def store(tmp_path: Path):
    config = AppConfig(data_dir=tmp_path / "data")
    vstore = VectorStore(config)
    vstore.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    yield vstore
    vstore.close()


def test_ensure_collections_idempotent_and_count(store: VectorStore) -> None:
    store.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    assert store.count() == 0
    assert store.count(CollectionName.VHDL) == 0
    assert store.count(CollectionName.DOCS) == 0
    assert store.count(CollectionName.CODE) == 0


def test_ensure_collection_dimension_drift(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    store.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    with pytest.raises(VectorStoreError, match="dense vector size"):
        store.ensure_collections(vhdl_dim=8, docs_dim=4, code_dim=4)
    store.close()


def test_point_id_deterministic_and_stable_across_commits() -> None:
    a = make_chunk()
    b = make_chunk(commit="def456")  # same content, new commit
    assert point_id(a) == point_id(b)
    c = make_chunk(symbol="p_read", start=21, end=30)
    assert point_id(a) != point_id(c)
    d = make_chunk(repo="repoB")
    assert point_id(a) != point_id(d)
    e = make_chunk(file="rtl/other.vhd")
    assert point_id(a) != point_id(e)
    f = make_chunk(
        collection=CollectionName.DOCS,
        content_type=ContentType.DOCUMENTATION,
        language="markdown",
    )
    assert point_id(a) != point_id(f)


def test_upsert_across_collections_and_query(store: VectorStore) -> None:
    chunks = [
        make_chunk(symbol="p_write", content="write process"),
        make_chunk(symbol="p_read", kind="process", start=21, end=30),
        make_chunk(
            repo="repoB",
            file="rtl/other.vhd",
            symbol="top",
            kind="entity",
        ),
        make_chunk(
            collection=CollectionName.CODE,
            content_type=ContentType.CODE,
            language="c",
            file="src/fifo.c",
            symbol="fifo_write",
            kind="function",
        ),
    ]
    store.upsert_chunks(
        chunks,
        dense=[dense(0), dense(1), dense(2), dense(3)],
        sparse=[sparse(), sparse(), sparse(), sparse()],
    )
    assert store.count() == 4
    assert store.count(CollectionName.VHDL) == 3
    assert store.count(CollectionName.CODE) == 1
    results = store.query(
        CollectionName.VHDL, dense=dense(2), sparse=sparse(), limit=10
    )
    assert [r.chunk.symbol for r in results] == ["top", "p_read", "p_write"]


def test_upsert_mismatched_lengths_raise(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError, match="1 chunks, 2 dense"):
        store.upsert_chunks(
            [make_chunk()], dense=[dense(0), dense(1)], sparse=[sparse(), sparse()]
        )
    with pytest.raises(VectorStoreError, match="1 chunks, 1 dense vectors, 2"):
        store.upsert_chunks(
            [make_chunk()], dense=[dense(0)], sparse=[sparse(), sparse()]
        )


def test_upsert_is_idempotent(store: VectorStore) -> None:
    store.upsert_chunks([make_chunk()], dense=[dense(0)], sparse=[sparse()])
    store.upsert_chunks([make_chunk()], dense=[dense(0)], sparse=[sparse()])
    assert store.count() == 1


def test_query_payload_filters(store: VectorStore) -> None:
    chunks = [
        make_chunk(symbol="p_write", entity="fifo", architecture="rtl"),
        make_chunk(repo="repoB", symbol="p_write", kind="process", start=21, end=30),
        make_chunk(
            collection=CollectionName.CODE,
            content_type=ContentType.CODE,
            language="c",
            file="src/fifo.c",
            symbol="fifo_write",
            kind="function",
        ),
    ]
    store.upsert_chunks(
        chunks,
        dense=[dense(0), dense(1), dense(2)],
        sparse=[sparse(), sparse(), sparse()],
    )
    by_repo = store.query(
        CollectionName.VHDL,
        dense=dense(0),
        sparse=sparse(),
        limit=10,
        must={"repository": "repoB"},
    )
    assert [r.chunk.symbol for r in by_repo] == ["p_write"]
    assert by_repo[0].chunk.repository == "repoB"
    by_entity = store.query(
        CollectionName.VHDL,
        dense=dense(0),
        sparse=sparse(),
        limit=10,
        must={"entity": "fifo"},
    )
    assert [r.chunk.symbol for r in by_entity] == ["p_write"]
    by_lang = store.query(
        CollectionName.CODE,
        dense=dense(0),
        sparse=sparse(),
        limit=10,
        must={"language": "c"},
    )
    assert [r.chunk.symbol for r in by_lang] == ["fifo_write"]
    assert (
        store.query(
            CollectionName.VHDL,
            dense=dense(0),
            sparse=sparse(),
            limit=10,
            must={"repository": "nope"},
        )
        == []
    )


def test_query_unknown_filter_key_raises(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError, match="unknown filter key"):
        store.query(
            CollectionName.VHDL,
            dense=dense(0),
            sparse=sparse(),
            limit=5,
            must={"not_a_key": "x"},
        )


def test_should_filter_matches_symbols_list(store: VectorStore) -> None:
    """The cross-referencing path: `should` OR-matches the symbols list field."""
    chunks = [
        make_chunk(
            collection=CollectionName.DOCS,
            content_type=ContentType.DOCUMENTATION,
            language="markdown",
            file="docs/standard.md",
            symbol="Reset conventions",
            kind="section",
            heading="Reset conventions",
            section="Standard",
            symbols=("p_write", "rst_n"),
        ),
        make_chunk(
            collection=CollectionName.DOCS,
            content_type=ContentType.DOCUMENTATION,
            language="markdown",
            file="docs/standard.md",
            symbol="Naming",
            kind="section",
            start=30,
            end=40,
            heading="Naming",
            section="Standard",
            symbols=("C_TIMEOUT",),
        ),
    ]
    store.upsert_chunks(chunks, dense=[dense(0), dense(1)], sparse=[sparse(), sparse()])
    hits = store.query(
        CollectionName.DOCS,
        dense=dense(0),
        sparse=sparse(),
        limit=10,
        should={"symbols": ["p_write"]},
    )
    assert [r.chunk.symbol for r in hits] == ["Reset conventions"]
    # OR across multiple candidate symbols
    hits = store.query(
        CollectionName.DOCS,
        dense=dense(0),
        sparse=sparse(),
        limit=10,
        should={"symbols": ["C_TIMEOUT", "nope"]},
    )
    assert [r.chunk.symbol for r in hits] == ["Naming"]


def test_sparse_leg_influences_ranking(store: VectorStore) -> None:
    """When the dense legs are a tie, the sparse (BM25) leg decides order."""
    chunks = [
        make_chunk(symbol="p_write", content="a"),
        make_chunk(symbol="p_read", kind="process", start=21, end=30, content="b"),
    ]
    # Identical dense vectors -> dense leg cannot discriminate.
    store.upsert_chunks(
        chunks,
        dense=[dense(0), dense(0)],
        sparse=[sparse(indices=(7, 8)), sparse(indices=(9,))],
    )
    # Sparse query matches only p_write's tokens.
    results = store.query(
        CollectionName.VHDL,
        dense=dense(1),  # orthogonal to passages: cosine 0 for both
        sparse=sparse(indices=(7,)),
        limit=10,
    )
    assert results, "expected results"
    assert results[0].chunk.symbol == "p_write"


def test_empty_sparse_falls_back_to_dense(store: VectorStore) -> None:
    store.upsert_chunks([make_chunk()], dense=[dense(0)], sparse=[sparse()])
    results = store.query(
        CollectionName.VHDL,
        dense=dense(0),
        sparse=SparseVectorData(indices=(), values=()),
        limit=10,
    )
    assert [r.chunk.symbol for r in results] == ["p_write"]


def test_delete_file_across_collections(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write"),
            make_chunk(symbol="p_read", kind="process", start=21, end=30),
            make_chunk(
                collection=CollectionName.DOCS,
                content_type=ContentType.DOCUMENTATION,
                language="markdown",
                file="docs/standard.md",
                symbol="FIFO",
                kind="section",
            ),
        ],
        dense=[dense(0), dense(1), dense(2)],
        sparse=[sparse(), sparse(), sparse()],
    )
    assert store.delete_file("repoA", "rtl/fifo.vhd") == 2
    assert store.count() == 1
    remaining = store.query(
        CollectionName.DOCS, dense=dense(2), sparse=sparse(), limit=10
    )
    assert remaining[0].chunk.symbol == "FIFO"
    assert store.delete_file("repoA", "missing.vhd") == 0


def test_delete_repository(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write"),
            make_chunk(
                repo="repoB",
                file="rtl/other.vhd",
                symbol="top",
                kind="entity",
            ),
        ],
        dense=[dense(0), dense(1)],
        sparse=[sparse(), sparse()],
    )
    assert store.delete_repository("repoA") == 1
    assert store.delete_repository("repoB") == 1
    assert store.count() == 0


def test_chunks_for_file_spans_collections(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write", entity="fifo"),
            make_chunk(
                collection=CollectionName.CODE,
                content_type=ContentType.CODE,
                language="python",
                file="tb/test_fifo.py",
                symbol="test_fifo",
                kind="function",
            ),
        ],
        dense=[dense(0), dense(1)],
        sparse=[sparse(), sparse()],
    )
    vhdl_file = store.chunks_for_file("repoA", "rtl/fifo.vhd")
    assert [c.symbol for c in vhdl_file] == ["p_write"]
    py_file = store.chunks_for_file("repoA", "tb/test_fifo.py")
    assert [c.symbol for c in py_file] == ["test_fifo"]
    assert py_file[0].collection is CollectionName.CODE
    assert store.chunks_for_file("repoA", "other.vhd") == []


def test_get_by_symbol(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write", entity="fifo"),
            make_chunk(
                symbol="p_read", kind="process", start=21, end=30, entity="fifo"
            ),
            make_chunk(symbol="log2", kind="function", start=31, end=40),
        ],
        dense=[dense(0), dense(1), dense(2)],
        sparse=[sparse(), sparse(), sparse()],
    )
    by_symbol = store.get_by_symbol("repoA", "log2", "function")
    assert [c.symbol for c in by_symbol] == ["log2"]
    assert store.get_by_symbol("repoA", "log2", "process") == []


def test_payload_roundtrip_content_and_attribution(store: VectorStore) -> None:
    chunk = make_chunk(
        symbol="p_write",
        entity="fifo",
        architecture="rtl",
        library="work",
        symbols=("wr_ptr", "C_FIFO_DEPTH"),
        content="p_write : process (clk, rst_n)\nbegin\nend process;\n",
    )
    store.upsert_chunks([chunk], dense=[dense(0)], sparse=[sparse()])
    results = store.query(CollectionName.VHDL, dense=dense(0), sparse=sparse(), limit=1)
    assert len(results) == 1
    got = results[0].chunk
    assert got.content == chunk.content
    assert got.symbols == ("wr_ptr", "C_FIFO_DEPTH")
    assert (
        got.repository,
        got.repository_category,
        got.repository_priority,
        got.branch,
        got.commit,
        got.file,
        got.symbol,
        got.symbol_kind,
        got.start_line,
        got.end_line,
    ) == (
        "repoA",
        "approved",
        90,
        "main",
        "abc123",
        "rtl/fifo.vhd",
        "p_write",
        "process",
        10,
        20,
    )
    assert (got.library, got.entity, got.architecture) == ("work", "fifo", "rtl")
    sr = SearchResult(
        result_type="vhdl",
        repository=got.repository,
        repository_category=got.repository_category,
        repository_priority=got.repository_priority,
        commit=got.commit,
        file=got.file,
        content=got.content,
        store_score=0.9,
        final_score=0.91,
        symbol=got.symbol,
        symbol_kind=got.symbol_kind,
        start_line=got.start_line,
        end_line=got.end_line,
        entity=got.entity,
        architecture=got.architecture,
        symbols=got.symbols,
    )
    md = sr.render()
    assert "repoA:rtl/fifo.vhd:10-20" in md
    assert "entity fifo" in md and "process p_write" in md
    assert "commit abc123" in md
    assert "wr_ptr, C_FIFO_DEPTH" in md


def test_persistence_across_reopen(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    store.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    store.upsert_chunks(
        [make_chunk(symbol="p_write")], dense=[dense(0)], sparse=[sparse()]
    )
    store.close()

    reopened = VectorStore(config)
    reopened.ensure_collections(vhdl_dim=4, docs_dim=4, code_dim=4)
    assert reopened.count() == 1
    assert (
        reopened.query(CollectionName.VHDL, dense=dense(0), sparse=sparse(), limit=5)[
            0
        ].chunk.symbol
        == "p_write"
    )
    reopened.close()


def test_queries_on_missing_collection_are_safe(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    assert (
        store.query(CollectionName.VHDL, dense=dense(0), sparse=sparse(), limit=5) == []
    )
    assert store.count() == 0
    assert store.delete_file("repoA", "f.vhd") == 0
    store.close()
