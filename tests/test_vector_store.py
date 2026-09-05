"""Tests for the SQLite/sqlite-vec vector store (embedded, temp dirs)."""

from __future__ import annotations

from pathlib import Path

import pytest
from capability import sqlite_extensions_supported

from corvidex_mcp.config import AppConfig
from corvidex_mcp.models import (
    INDEX_SCHEMA_VERSION,
    Chunk,
    CollectionName,
    ContentType,
    SearchResult,
)
from corvidex_mcp.vector_store import VectorStore, VectorStoreError, point_id

pytestmark = pytest.mark.skipif(
    not sqlite_extensions_supported(),
    reason=(
        "stdlib SQLite lacks loadable-extension support (the sqlite-vec "
        "extension cannot load; use CPython 3.14 or a system/homebrew "
        "Python)"
    ),
)


def make_chunk(
    repo: str = "repoA",
    file: str = "rtl/fifo.vhd",
    symbol: str = "p_write",
    kind: str = "process",
    start: int = 10,
    end: int = 20,
    content: str = "process body",
    collection: CollectionName = CollectionName.HDL,
    content_type: ContentType = ContentType.SOURCE,
    language: str = "vhdl",
    symbols: tuple[str, ...] = (),
    **extra: object,
) -> Chunk:
    return Chunk(
        repository=repo,
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
        module=extra.get("module"),
        native_symbol_kind=extra.get("native_symbol_kind"),
        heading=extra.get("heading"),
        section=extra.get("section"),
        symbols=symbols,
    )


def dense(i: int) -> list[float]:
    return [float(i), 0.0, 0.0, 0.0]


@pytest.fixture
def store(tmp_path: Path):
    config = AppConfig(data_dir=tmp_path / "data")
    vstore = VectorStore(config)
    vstore.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    yield vstore
    vstore.close()


def test_ensure_collections_idempotent_and_count(store: VectorStore) -> None:
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    assert store.count() == 0
    assert store.count(CollectionName.HDL) == 0
    assert store.count(CollectionName.DOCS) == 0
    assert store.count(CollectionName.CODE) == 0


def test_ensure_collection_dimension_drift(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    with pytest.raises(VectorStoreError, match="dense vector size"):
        store.ensure_collections(hdl_dim=8, docs_dim=4, code_dim=4)
    store.close()


def test_schema_version_current_on_fresh_store(store: VectorStore) -> None:
    # A fresh, ensured store reports the current version (inferred from
    # its tables); migrate() stamps it without modifying data.
    assert store.schema_version == INDEX_SCHEMA_VERSION
    assert store.migrate() is False
    assert store._read_version() == INDEX_SCHEMA_VERSION  # stamped
    assert store.migrate() is False  # idempotent


def test_migration_v1_to_v2_drops_the_legacy_collection(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    # A v1 deployment: only the legacy ``vhdl`` collection, no meta row.
    store._conn.execute("CREATE TABLE chunks_vhdl (id TEXT PRIMARY KEY)")
    store._conn.execute("CREATE VIRTUAL TABLE vec_vhdl USING vec0(embedding float[4])")
    store._conn.execute("CREATE VIRTUAL TABLE fts_vhdl USING fts5(content)")
    store._conn.commit()
    assert store.schema_version == 1
    assert store.migrate() is True
    assert "vhdl" not in store._tables()
    assert store.schema_version == INDEX_SCHEMA_VERSION
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    assert {"hdl", "docs", "code"} <= store._tables()
    assert store.migrate() is False  # idempotent
    store.close()


def test_migration_reinfers_version_when_meta_is_missing(
    store: VectorStore,
) -> None:
    store.migrate()  # stamps the meta row
    store._conn.execute("DELETE FROM meta")
    assert store._read_version() is None
    assert store.schema_version == INDEX_SCHEMA_VERSION  # inferred
    assert store.migrate() is False  # re-stamp only, no data change
    assert store._read_version() == INDEX_SCHEMA_VERSION


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
        make_chunk(
            symbol="p_read",
            kind="process",
            start=21,
            end=30,
            content="read process",
        ),
        make_chunk(
            repo="repoB",
            file="rtl/other.vhd",
            symbol="top",
            kind="entity",
            content="top entity",
        ),
        make_chunk(
            collection=CollectionName.CODE,
            content_type=ContentType.CODE,
            language="c",
            file="src/fifo.c",
            symbol="fifo_write",
            kind="function",
            content="c fifo function",
        ),
    ]
    store.upsert_chunks(
        chunks,
        dense=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert store.count() == 4
    assert store.count(CollectionName.HDL) == 3
    assert store.count(CollectionName.CODE) == 1
    results = store.query(
        CollectionName.HDL,
        dense=[0.0, 0.0, 1.0, 0.0],
        query_text="top",
        limit=10,
    )
    assert results[0].chunk.symbol == "top"


def test_count_repository(store: VectorStore) -> None:
    chunks = [
        make_chunk(symbol="p_write"),
        make_chunk(symbol="p_read", kind="process", start=21, end=30),
        make_chunk(repo="repoB", file="rtl/other.vhd", symbol="top", kind="entity"),
        make_chunk(
            repo="repoB",
            collection=CollectionName.CODE,
            content_type=ContentType.CODE,
            language="c",
            file="src/fifo.c",
            symbol="fifo_write",
            kind="function",
        ),
    ]
    store.upsert_chunks(chunks, dense=[dense(0), dense(1), dense(2), dense(3)])
    assert store.count_repository("repoA") == 2
    assert store.count_repository("repoB") == 2
    assert store.count_repository("repoB", CollectionName.HDL) == 1
    assert store.count_repository("repoB", CollectionName.CODE) == 1
    assert store.count_repository("repoA", CollectionName.CODE) == 0
    assert store.count_repository("unknown") == 0


def test_list_files(store: VectorStore) -> None:
    chunks = [
        make_chunk(symbol="p_write"),
        make_chunk(symbol="p_read", kind="process", start=21, end=30),
        make_chunk(repo="repoB", file="rtl/other.vhd", symbol="top", kind="entity"),
        make_chunk(
            repo="repoB",
            collection=CollectionName.CODE,
            content_type=ContentType.CODE,
            language="c",
            file="src/fifo.c",
            symbol="fifo_write",
            kind="function",
        ),
    ]
    store.upsert_chunks(chunks, dense=[dense(0), dense(1), dense(2), dense(3)])
    assert store.list_files("repoA") == ["rtl/fifo.vhd"]
    assert store.list_files("repoB") == ["rtl/other.vhd", "src/fifo.c"]
    assert store.list_files("repoB", CollectionName.HDL) == ["rtl/other.vhd"]
    assert store.list_files("unknown") == []


def test_upsert_mismatched_lengths_raise(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError, match="1 chunks, 2 dense"):
        store.upsert_chunks([make_chunk()], dense=[dense(0), dense(1)])


def test_upsert_is_idempotent(store: VectorStore) -> None:
    store.upsert_chunks([make_chunk()], dense=[dense(1)])
    store.upsert_chunks([make_chunk()], dense=[dense(1)])
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
    store.upsert_chunks(chunks, dense=[dense(1), dense(1), dense(1)])
    by_repo = store.query(
        CollectionName.HDL,
        dense=dense(1),
        query_text="process",
        limit=10,
        must={"repository": "repoB"},
    )
    assert [r.chunk.symbol for r in by_repo] == ["p_write"]
    assert by_repo[0].chunk.repository == "repoB"
    by_entity = store.query(
        CollectionName.HDL,
        dense=dense(1),
        query_text="process",
        limit=10,
        must={"entity": "fifo"},
    )
    assert [r.chunk.symbol for r in by_entity] == ["p_write"]
    by_lang = store.query(
        CollectionName.CODE,
        dense=dense(1),
        query_text="function",
        limit=10,
        must={"language": "c"},
    )
    assert [r.chunk.symbol for r in by_lang] == ["fifo_write"]
    assert (
        store.query(
            CollectionName.HDL,
            dense=dense(1),
            query_text="process",
            limit=10,
            must={"repository": "nope"},
        )
        == []
    )


def test_query_unknown_filter_key_raises(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError, match="unknown filter key"):
        store.query(
            CollectionName.HDL,
            dense=dense(1),
            query_text="process",
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
            content="reset conventions rst_n",
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
            content="naming timeout",
        ),
    ]
    store.upsert_chunks(chunks, dense=[dense(1), dense(2)])
    hits = store.query(
        CollectionName.DOCS,
        dense=dense(1),
        query_text="reset",
        limit=10,
        should={"symbols": ["p_write"]},
    )
    assert [r.chunk.symbol for r in hits] == ["Reset conventions"]
    # OR across multiple candidate symbols
    hits = store.query(
        CollectionName.DOCS,
        dense=dense(1),
        query_text="naming",
        limit=10,
        should={"symbols": ["C_TIMEOUT", "nope"]},
    )
    assert [r.chunk.symbol for r in hits] == ["Naming"]


def test_fts_leg_influences_ranking(store: VectorStore) -> None:
    """When the dense legs are a tie, the full-text leg decides order."""
    chunks = [
        make_chunk(symbol="p_write", content="alpha token"),
        make_chunk(
            symbol="p_read",
            kind="process",
            start=21,
            end=30,
            content="beta token",
        ),
    ]
    # Identical dense vectors -> the dense leg cannot discriminate.
    store.upsert_chunks(chunks, dense=[dense(1), dense(1)])
    # Dense query is orthogonal to both passages (cosine 0 for both); the
    # full-text query matches only p_write's content.
    results = store.query(
        CollectionName.HDL,
        dense=[0.0, 1.0, 0.0, 0.0],
        query_text="alpha",
        limit=10,
    )
    assert results, "expected results"
    assert results[0].chunk.symbol == "p_write"


def test_empty_query_text_falls_back_to_dense(store: VectorStore) -> None:
    store.upsert_chunks([make_chunk(content="write process")], dense=[dense(1)])
    results = store.query(
        CollectionName.HDL,
        dense=dense(1),
        query_text="",
        limit=10,
    )
    assert [r.chunk.symbol for r in results] == ["p_write"]


def test_delete_file_across_collections(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write", content="write process"),
            make_chunk(
                symbol="p_read",
                kind="process",
                start=21,
                end=30,
                content="read process",
            ),
            make_chunk(
                collection=CollectionName.DOCS,
                content_type=ContentType.DOCUMENTATION,
                language="markdown",
                file="docs/standard.md",
                symbol="FIFO",
                kind="section",
                content="fifo documentation",
            ),
        ],
        dense=[dense(1), dense(2), dense(3)],
    )
    assert store.delete_file("repoA", "rtl/fifo.vhd") == 2
    assert store.count() == 1
    remaining = store.query(
        CollectionName.DOCS, dense=dense(3), query_text="documentation", limit=10
    )
    assert remaining[0].chunk.symbol == "FIFO"
    assert store.delete_file("repoA", "missing.vhd") == 0


def test_delete_repository(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write", content="write process"),
            make_chunk(
                repo="repoB",
                file="rtl/other.vhd",
                symbol="top",
                kind="entity",
                content="top entity",
            ),
        ],
        dense=[dense(1), dense(2)],
    )
    assert store.delete_repository("repoA") == 1
    assert store.delete_repository("repoB") == 1
    assert store.count() == 0


def test_chunks_for_file_spans_collections(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(symbol="p_write", entity="fifo", content="write process"),
            make_chunk(
                collection=CollectionName.CODE,
                content_type=ContentType.CODE,
                language="python",
                file="tb/test_fifo.py",
                symbol="test_fifo",
                kind="function",
                content="test fifo python",
            ),
        ],
        dense=[dense(1), dense(2)],
    )
    vhdl_file = store.chunks_for_file("repoA", "rtl/fifo.vhd")
    assert [c.symbol for c in vhdl_file] == ["p_write"]
    py_file = store.chunks_for_file("repoA", "tb/test_fifo.py")
    assert [c.symbol for c in py_file] == ["test_fifo"]
    assert py_file[0].collection is CollectionName.CODE
    assert store.chunks_for_file("repoA", "other.vhd") == []


def test_payload_roundtrip_content_and_attribution(store: VectorStore) -> None:
    chunk = make_chunk(
        symbol="p_write",
        entity="fifo",
        architecture="rtl",
        library="work",
        module="fifo",
        native_symbol_kind="entity",
        symbols=("wr_ptr", "C_FIFO_DEPTH"),
        content="p_write : process (clk, rst_n)\nbegin\nend process;\n",
    )
    store.upsert_chunks([chunk], dense=[dense(1)])
    results = store.query(
        CollectionName.HDL, dense=dense(1), query_text="p_write process", limit=1
    )
    assert len(results) == 1
    got = results[0].chunk
    assert got.content == chunk.content
    assert got.symbols == ("wr_ptr", "C_FIFO_DEPTH")
    assert got.language == "vhdl"
    assert (got.module, got.native_symbol_kind) == ("fifo", "entity")
    assert (
        got.repository,
        got.branch,
        got.commit,
        got.file,
        got.symbol,
        got.symbol_kind,
        got.start_line,
        got.end_line,
    ) == (
        "repoA",
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
        result_type="hdl",
        repository=got.repository,
        commit=got.commit,
        file=got.file,
        content=got.content,
        score=0.9,
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
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    store.upsert_chunks(
        [make_chunk(symbol="p_write", content="write process")], dense=[dense(1)]
    )
    store.close()

    reopened = VectorStore(config)
    reopened.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    assert reopened.count() == 1
    assert (
        reopened.query(
            CollectionName.HDL, dense=dense(1), query_text="write process", limit=5
        )[0].chunk.symbol
        == "p_write"
    )
    reopened.close()


def test_queries_on_missing_collection_are_safe(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    assert (
        store.query(CollectionName.HDL, dense=dense(1), query_text="process", limit=5)
        == []
    )
    assert store.count() == 0
    assert store.delete_file("repoA", "f.vhd") == 0
    store.close()


# -- batched writes ----------------------------------------------------------


def _row_count(store: VectorStore, table: str) -> int:
    row = store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0])


def test_delete_files_batches_across_the_bound_parameter_limit(
    store: VectorStore,
) -> None:
    """>900 files in one call: the IN-list is chunked, one transaction."""
    n = 1200
    files = [f"rtl/f{i}.vhd" for i in range(n)]
    chunks = [make_chunk(file=f, symbol=f"s{i}") for i, f in enumerate(files)]
    keep = make_chunk(file="rtl/keep.vhd", symbol="keep")
    store.upsert_chunks([*chunks, keep], dense=[dense(i) for i in range(n + 1)])
    assert store.count() == n + 1

    deleted = store.delete_files("repoA", files)
    assert deleted == n
    assert store.count() == 1
    assert store.chunks_for_file("repoA", "rtl/keep.vhd") != []
    for f in (files[0], files[n // 2], files[-1]):
        assert store.chunks_for_file("repoA", f) == []
    # Rows are gone from every underlying table, not just chunks_hdl.
    assert _row_count(store, "vec_hdl") == 1
    assert _row_count(store, "fts_hdl") == 1


def test_delete_files_spans_collections_in_one_call(store: VectorStore) -> None:
    store.upsert_chunks(
        [
            make_chunk(file="rtl/fifo.vhd", symbol="p_write"),
            make_chunk(
                collection=CollectionName.DOCS,
                content_type=ContentType.DOCUMENTATION,
                language="markdown",
                file="docs/standard.md",
                symbol="FIFO",
                kind="section",
                content="fifo documentation",
            ),
            make_chunk(
                collection=CollectionName.CODE,
                content_type=ContentType.CODE,
                language="c",
                file="src/fifo.c",
                symbol="fifo_write",
                kind="function",
            ),
        ],
        dense=[dense(0), dense(1), dense(2)],
    )
    deleted = store.delete_files(
        "repoA", ["rtl/fifo.vhd", "docs/standard.md", "src/fifo.c"]
    )
    assert deleted == 3
    assert store.count() == 0


def test_delete_files_empty_is_a_no_op(store: VectorStore) -> None:
    store.upsert_chunks([make_chunk()], dense=[dense(1)])
    assert store.delete_files("repoA", []) == 0
    assert store.count() == 1


def test_upsert_more_than_the_bound_parameter_limit(store: VectorStore) -> None:
    """>900 chunks in one call: the id-lookup and inserts are chunked. Every
    ``dense(i)`` for i >= 1 normalizes to the same unit vector, so identity
    is checked via row data (not vector similarity, which cannot
    distinguish them) for a chunk from each end of the batch split."""
    n = 1200
    chunks = [make_chunk(file=f"rtl/f{i}.vhd", symbol=f"s{i}") for i in range(n)]
    store.upsert_chunks(chunks, dense=[dense(i) for i in range(n)])
    assert store.count(CollectionName.HDL) == n
    assert _row_count(store, "vec_hdl") == n
    assert _row_count(store, "fts_hdl") == n
    for i in (0, 500, 899, 900, n - 1):
        chunks_for_file = store.chunks_for_file("repoA", f"rtl/f{i}.vhd")
        assert [c.symbol for c in chunks_for_file] == [f"s{i}"]


def test_upsert_on_conflict_replaces_vec_and_fts_rows_without_duplicates(
    store: VectorStore,
) -> None:
    """Re-upserting the same chunk (across separate calls) hits the
    ON CONFLICT UPDATE path; the old vec/fts rows must not linger."""
    chunk = make_chunk(content="original body")
    store.upsert_chunks([chunk], dense=[dense(1)])
    assert store.count(CollectionName.HDL) == 1
    assert _row_count(store, "vec_hdl") == 1
    assert _row_count(store, "fts_hdl") == 1

    updated = make_chunk(content="rewritten body")
    assert point_id(updated) == point_id(chunk)  # same canonical id
    store.upsert_chunks([updated], dense=[dense(2)])

    assert store.count(CollectionName.HDL) == 1
    assert _row_count(store, "vec_hdl") == 1
    assert _row_count(store, "fts_hdl") == 1
    # Lexical mode isolates the full-text leg: hybrid mode would still
    # surface this chunk for the "original" query via its dense leg (it
    # is the only row in the collection), which would not tell us
    # whether the old fts5 row was actually replaced.
    results = store.query(
        CollectionName.HDL,
        dense=dense(2),
        query_text="rewritten",
        limit=5,
        mode="lexical",
    )
    assert [r.chunk.content for r in results] == ["rewritten body"]
    stale = store.query(
        CollectionName.HDL,
        dense=dense(2),
        query_text="original",
        limit=5,
        mode="lexical",
    )
    assert stale == []


def test_upsert_duplicate_id_within_one_call_last_write_wins(
    store: VectorStore,
) -> None:
    first = make_chunk(content="first")
    second = make_chunk(content="second")  # same repo/file/symbol/lines -> same id
    assert point_id(first) == point_id(second)
    store.upsert_chunks([first, second], dense=[dense(1), dense(2)])
    assert store.count(CollectionName.HDL) == 1
    assert _row_count(store, "vec_hdl") == 1
    assert _row_count(store, "fts_hdl") == 1
    results = store.query(
        CollectionName.HDL, dense=dense(2), query_text="second", limit=5
    )
    assert [r.chunk.content for r in results] == ["second"]


def test_index_drop_migration_on_an_existing_db(tmp_path: Path) -> None:
    """A database created before the redundant-index cleanup has its
    ``idx_<c>_repo`` and ``idx_<c>_symbol`` indexes dropped on ensure."""
    config = AppConfig(data_dir=tmp_path / "data")
    legacy = VectorStore(config)
    legacy._conn.execute(
        'CREATE TABLE chunks_hdl (id TEXT PRIMARY KEY, "repository" TEXT, '
        '"file" TEXT, "symbol_kind" TEXT)'
    )
    legacy._conn.execute('CREATE INDEX idx_hdl_repo ON chunks_hdl("repository")')
    legacy._conn.execute(
        'CREATE INDEX idx_hdl_repo_file ON chunks_hdl("repository", "file")'
    )
    legacy._conn.execute(
        'CREATE INDEX idx_hdl_symbol ON chunks_hdl("symbol_kind", "repository")'
    )
    legacy._conn.execute("CREATE VIRTUAL TABLE vec_hdl USING vec0(embedding float[4])")
    legacy._conn.execute("CREATE VIRTUAL TABLE fts_hdl USING fts5(content)")
    legacy._conn.commit()
    legacy.close()

    def index_names() -> set[str]:
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'chunks_hdl'"
        ).fetchall()
        return {row[0] for row in rows}

    store = VectorStore(config)
    assert {"idx_hdl_repo", "idx_hdl_repo_file", "idx_hdl_symbol"} <= index_names()
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    # The PRIMARY KEY's implicit unique index is untouched — only the two
    # redundant secondary indexes are dropped.
    assert index_names() == {"idx_hdl_repo_file", "sqlite_autoindex_chunks_hdl_1"}
    store.close()


def test_serialize_vector_is_byte_identical_to_sqlite_vec() -> None:
    import numpy as np
    import sqlite_vec

    from corvidex_mcp.vector_store import _serialize_vector, _unit_vector

    rng = np.random.default_rng(42)
    raw = rng.standard_normal(32).astype(np.float32)
    unit = _unit_vector(raw.tolist())
    assert _serialize_vector(unit) == sqlite_vec.serialize_float32(unit.tolist())
