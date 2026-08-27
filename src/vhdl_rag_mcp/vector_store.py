"""LanceDB-backed vector store — the only module that knows LanceDB.

Design notes
------------
- One embedded LanceDB instance under ``<data_dir>/lancedb`` by default.
  No separate server process is required; data is a set of columnar
  Lance table files.
- Three tables in that single instance:
    ``hdl``   — semantically chunked HDL (VHDL, Verilog, SystemVerilog)
    ``docs``  — HDL-related documentation
    ``code``  — general source code (C/C++, Python, ...)
  Tables may use different dense embedding models; the store keeps them
  isolated. Upgrading from the v1 layout (a ``vhdl`` table) is a
  deterministic full reindex; :meth:`VectorStore.delete_legacy_vhdl`
  drops the old table safely because every chunk is reproducible from
  the git history.
- Every table carries one dense vector column (``dense``, fixed-size
  float32) plus the chunk metadata. Search is a native LanceDB hybrid
  query: the dense vector leg is fused with a full-text (inverted)
  index over the chunk content, RRF-style, so results combine semantic
  similarity and exact identifier/token matching in one ranked list.
  The FTS leg replaces the BM25 sparse-vector leg of the Qdrant layout
  — and with it the separate BM25 embedding model (identifiers such as
  ``fifo_ctrl`` or ``rst_n`` match through the simple tokenizer, which
  splits on punctuation and ORs the tokens).
- Chunk rows are keyed by a deterministic ``uuid5`` id over
  :meth:`Chunk.canonical_id`, so re-upserts are idempotent merges
  (``merge_insert``); stale-chunk cleanup happens via SQL predicates
  (repository/file), never by recomputing IDs.
- Vector search is flat (exact) KNN: at repository scale (hundreds to
  tens of thousands of chunks) that is both fast and exact. If a
  deployment grows beyond that, an HNSW/IVF index can be added on the
  ``dense`` column without touching the rest of the store.
- The MCP layer only ever talks about repositories, files, symbols, and
  results — LanceDB terminology does not leak out of this module.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from lancedb.index import FTS
from lancedb.query import LanceQueryBuilder
from lancedb.table import Table

from .config import AppConfig
from .models import Chunk, CollectionName

logger = logging.getLogger(__name__)

COLLECTION_HDL = "hdl"
COLLECTION_DOCS = "docs"
COLLECTION_CODE = "code"
ALL_COLLECTIONS: tuple[CollectionName, ...] = (
    CollectionName.HDL,
    CollectionName.DOCS,
    CollectionName.CODE,
)

DENSE_COLUMN = "dense"
FTS_COLUMN = "content"
ID_COLUMN = "id"

# FTS configuration tuned for HDL source: the simple tokenizer splits
# on punctuation (so ``fifo_ctrl`` yields fifo/ctrl and a query for the
# same identifier ORs its tokens). Stemming is disabled so code
# identifiers survive tokenization verbatim; stop-word removal is
# enabled so common English words in a query (the, by, ...) do not pollute
# the exact-match leg — identifiers such as ``rst_n`` are never stop words.
_FTS_CONFIG = FTS(base_tokenizer="simple", stem=False, remove_stop_words=True)

# Namespace for deterministic row IDs (stable across commits and
# processes; makes re-upserts idempotent).
_ID_NAMESPACE = uuid.NAMESPACE_URL
_ID_PREFIX = "vhdl-rag-mcp::"

# Metadata columns (everything :meth:`Chunk.payload` stores, plus the id
# and dense vector).
_PAYLOAD_FIELDS: tuple[str, ...] = (
    "repository",
    "branch",
    "commit",
    "file",
    "content_type",
    "language",
    "collection",
    "symbol",
    "symbol_kind",
    "start_line",
    "end_line",
    "content",
    "library",
    "entity",
    "architecture",
    "module",
    "native_symbol_kind",
    "heading",
    "section",
    "symbols",
)

# Metadata fields that may be used as equality filters in queries.
_FILTER_KEYS = frozenset(
    {
        "repository",
        "file",
        "symbol_kind",
        "symbol",
        "entity",
        "architecture",
        "heading",
        "section",
        "language",
        "symbols",
    }
)

#: Rows returned by get_by_symbol are capped (a symbol rarely appears in
#: more than a few constructs of one kind).
_SYMBOL_LOOKUP_LIMIT = 64


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot perform an operation."""


def point_id(chunk: Chunk) -> str:
    """Deterministic UUIDv5 row ID for a chunk (stable across commits)."""
    return str(uuid.uuid5(_ID_NAMESPACE, _ID_PREFIX + chunk.canonical_id))


def _check_filter_key(key: str) -> None:
    if key not in _FILTER_KEYS:
        raise VectorStoreError(f"unknown filter key: {key!r}")


def _sql_quote(value: str) -> str:
    """SQL single-quoted literal (embedded quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


def _where_clause(
    must: Mapping[str, str] | None,
    should: Mapping[str, Sequence[str]] | None,
) -> str | None:
    """SQL predicate: AND of ``must`` equalities plus optional OR groups.

    ``should`` maps a column to a list of values: the row matches if the
    column contains ANY of them (``array_contains`` — this is how the
    cross-referencing ``symbols`` list field is queried).
    """
    parts: list[str] = []
    if must:
        for key, value in must.items():
            _check_filter_key(key)
            parts.append(f"{key} = {_sql_quote(value)}")
    if should:
        for key, values in should.items():
            _check_filter_key(key)
            parts.append(
                "("
                + " OR ".join(
                    f"array_contains({key}, {_sql_quote(value)})" for value in values
                )
                + ")"
            )
    return " AND ".join(parts) if parts else None


def _table_schema(dim: int) -> pa.Schema:
    """Schema for one collection table: metadata + fixed-size float32 vector."""
    return pa.schema(
        [
            (ID_COLUMN, pa.string()),
            ("repository", pa.string()),
            ("branch", pa.string()),
            ("commit", pa.string()),
            ("file", pa.string()),
            ("content_type", pa.string()),
            ("language", pa.string()),
            ("collection", pa.string()),
            ("symbol", pa.string()),
            ("symbol_kind", pa.string()),
            ("start_line", pa.int64()),
            ("end_line", pa.int64()),
            ("content", pa.string()),
            ("library", pa.string()),
            ("entity", pa.string()),
            ("architecture", pa.string()),
            ("module", pa.string()),
            ("native_symbol_kind", pa.string()),
            ("heading", pa.string()),
            ("section", pa.string()),
            ("symbols", pa.list_(pa.string())),
            (DENSE_COLUMN, pa.list_(pa.float32(), dim)),
        ]
    )


def _dense_dimension(tbl: Table) -> int | None:
    field = tbl.schema.field(DENSE_COLUMN)
    if isinstance(field.type, pa.FixedSizeListType):
        return int(field.type.list_size)
    return None


def _fts_indexed(name: str, db: lancedb.DBConnection) -> bool:
    """Whether the table has an inverted (FTS) index over ``content``.

    A freshly opened dataset is used: a dataset handle opened before the
    index was created reports a stale (empty) index list.
    """
    if name not in set(db.list_tables().tables):
        return False
    dataset = db.open_table(name).to_lance()
    describe = getattr(dataset, "describe_indices", None)
    if describe is None:  # pragma: no cover - older lance releases
        indices = dataset.list_indices()
        return any(
            ix.get("type") == "Inverted" and FTS_COLUMN in (ix.get("fields") or [])
            for ix in indices
        )
    return any(
        ix.index_type == "Inverted" and FTS_COLUMN in (ix.field_names or [])
        for ix in describe()
    )


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from its stored metadata row."""
    from .models import CollectionName, ContentType

    symbols_raw = payload.get("symbols") or []
    return Chunk(
        repository=payload["repository"],
        branch=payload["branch"],
        commit=payload["commit"],
        file=payload["file"],
        content_type=ContentType(payload["content_type"]),
        language=payload["language"],
        collection=CollectionName(payload["collection"]),
        symbol=payload["symbol"],
        symbol_kind=payload["symbol_kind"],
        start_line=payload["start_line"],
        end_line=payload["end_line"],
        content=payload["content"],
        library=payload.get("library"),
        entity=payload.get("entity"),
        architecture=payload.get("architecture"),
        module=payload.get("module"),
        native_symbol_kind=payload.get("native_symbol_kind"),
        heading=payload.get("heading"),
        section=payload.get("section"),
        symbols=tuple(symbols_raw),
    )


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its hybrid (RRF-fused) relevance score."""

    score: float
    chunk: Chunk


class VectorStore:
    """Thin, typed wrapper around a single embedded LanceDB instance."""

    def __init__(self, config: AppConfig) -> None:
        path = config.lance_local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))
        logger.info("lancedb embedded store: %s", path)

    def close(self) -> None:
        # Embedded store: nothing to release; the tables are plain files.
        pass

    # -- tables ---------------------------------------------------------

    def _tables(self) -> set[str]:
        return set(self._db.list_tables().tables)

    def _table(self, name: str) -> Table | None:
        if name not in self._tables():
            return None
        return self._db.open_table(name)

    def ensure_collections(self, hdl_dim: int, docs_dim: int, code_dim: int) -> None:
        """Create the tables if missing; fail loudly on dimension drift.

        A changed embedding model changes the dense vector size; silently
        mixing dimensions would corrupt retrieval, so this is an error
        with an actionable message.
        """
        for name, dim in (
            (COLLECTION_HDL, hdl_dim),
            (COLLECTION_DOCS, docs_dim),
            (COLLECTION_CODE, code_dim),
        ):
            tbl = self._table(name)
            if tbl is None:
                self._db.create_table(name, schema=_table_schema(dim))
                # FTS is mandatory for hybrid search (a search on a table
                # without the inverted index is a hard error); create it
                # up front, so even empty tables are queryable.
                self._db.open_table(name).create_index(FTS_COLUMN, config=_FTS_CONFIG)
                logger.info("created lancedb table %r (dim=%d)", name, dim)
                continue
            size = _dense_dimension(tbl)
            if size != dim:
                raise VectorStoreError(
                    f"table {name!r} has dense vector size {size}, "
                    f"expected {dim}. The embedding model changed — "
                    "delete the table (or the lancedb data directory) "
                    "and reindex."
                )
            # FTS is mandatory for hybrid search (a search on a table
            # without the inverted index is a hard error); create it on
            # pre-existing tables that lack it.
            if not _fts_indexed(name, self._db):
                self._db.open_table(name).create_index(FTS_COLUMN, config=_FTS_CONFIG)
                logger.info("created FTS index on existing table %r", name)

    def delete_legacy_vhdl(self) -> bool:
        """Drop the v1-layout ``vhdl`` table (safe: reproducible).

        Returns True when a legacy table was removed.
        """
        name = "vhdl"
        if name not in self._tables():
            return False
        self._db.drop_table(name)
        logger.info("deleted legacy lancedb table %r (v1 layout)", name)
        return True

    # -- writes ---------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], dense: list[list[float]]) -> None:
        """Upsert chunks with precomputed dense vectors (same order/length)."""
        if len(chunks) != len(dense):
            raise VectorStoreError(
                f"got {len(chunks)} chunks, {len(dense)} dense vectors"
            )
        by_collection: dict[str, list[tuple[Chunk, list[float]]]] = {}
        for chunk, d in zip(chunks, dense, strict=True):
            by_collection.setdefault(chunk.collection.value, []).append((chunk, d))
        for collection_name, items in by_collection.items():
            if not items:
                continue
            tbl = self._table(collection_name)
            if tbl is None:
                # Defensive: tables are created in ensure_collections.
                tbl = self._db.create_table(
                    collection_name, schema=_table_schema(len(items[0][1]))
                )
            rows = []
            for chunk, d in items:
                payload = chunk.payload()
                row: dict[str, Any] = {
                    ID_COLUMN: point_id(chunk),
                    DENSE_COLUMN: list(d),
                }
                for field in _PAYLOAD_FIELDS:
                    value = payload.get(field)
                    row[field] = [] if field == "symbols" and value is None else value
                rows.append(row)
            (
                tbl.merge_insert(ID_COLUMN)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            if not _fts_indexed(collection_name, self._db):
                tbl.create_index(FTS_COLUMN, config=_FTS_CONFIG)
                logger.info("created FTS index on table %r", collection_name)
            logger.info("upserted %d chunks into table %r", len(items), collection_name)

    def delete_file(self, repository: str, file: str) -> int:
        """Remove all chunks for one file from every collection."""
        total = 0
        for collection in ALL_COLLECTIONS:
            total += self._delete(
                collection, must={"repository": repository, "file": file}
            )
        return total

    def delete_repository(self, repository: str) -> int:
        total = 0
        for collection in ALL_COLLECTIONS:
            total += self._delete(collection, must={"repository": repository})
        return total

    def _delete(self, collection: CollectionName, must: Mapping[str, str]) -> int:
        name = collection.value
        tbl = self._table(name)
        if tbl is None:
            return 0
        before = tbl.count_rows()
        predicate = " AND ".join(
            f"{key} = {_sql_quote(value)}" for key, value in must.items()
        )
        tbl.delete(predicate)
        after = tbl.count_rows()
        deleted = before - after
        if deleted:
            logger.info(
                "deleted %d stale chunks from %r for %s",
                deleted,
                name,
                " ".join(f"{k}={v}" for k, v in must.items()),
            )
        return deleted

    # -- reads ----------------------------------------------------------

    def query(
        self,
        collection: CollectionName,
        dense: list[float],
        query_text: str,
        limit: int,
        must: Mapping[str, str] | None = None,
        should: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ScoredChunk]:
        """Hybrid (dense + full-text, RRF-fused) search with row filters.

        ``query_text`` feeds the full-text leg (the raw query string);
        an empty text falls back to the dense leg alone.
        """
        tbl = self._table(collection.value)
        if tbl is None:
            return []
        predicate = _where_clause(must, should)
        vector = np.asarray(dense, dtype=np.float32)
        text = (query_text or "").strip()
        builder: LanceQueryBuilder
        if text:
            builder = tbl.search(query_type="hybrid").vector(vector).text(text)
        else:
            builder = tbl.search(vector)
        if predicate is not None:
            builder = builder.where(predicate)
        rows = builder.limit(limit).to_list()
        return [
            ScoredChunk(score=_row_score(row), chunk=chunk_from_payload(row))
            for row in rows
        ]

    def count(self, collection: CollectionName | None = None) -> int:
        if collection is not None:
            tbl = self._table(collection.value)
            return tbl.count_rows() if tbl is not None else 0
        return sum(self.count(c) for c in ALL_COLLECTIONS)

    def count_repository(
        self, repository: str, collection: CollectionName | None = None
    ) -> int:
        """Rows of one repository, in one collection or all of them."""
        predicate = f"repository = {_sql_quote(repository)}"
        if collection is not None:
            tbl = self._table(collection.value)
            if tbl is None:
                return 0
            return tbl.to_lance().count_rows(filter=predicate)
        return sum(self.count_repository(repository, c) for c in ALL_COLLECTIONS)

    def chunks_for_file(self, repository: str, file: str) -> list[Chunk]:
        """All currently indexed chunks for one file (every collection)."""
        predicate = (
            f"repository = {_sql_quote(repository)} AND file = {_sql_quote(file)}"
        )
        chunks: list[Chunk] = []
        for collection in ALL_COLLECTIONS:
            tbl = self._table(collection.value)
            if tbl is None:
                continue
            table = tbl.to_lance().to_table(
                filter=predicate, columns=list(_PAYLOAD_FIELDS)
            )
            chunks.extend(chunk_from_payload(row) for row in table.to_pylist())
        return chunks

    def get_by_symbol(
        self, repository: str, symbol: str, symbol_kind: str
    ) -> list[Chunk]:
        """HDL chunks of one kind named ``symbol`` in ``repository``."""
        name = COLLECTION_HDL
        tbl = self._table(name)
        if tbl is None:
            return []
        predicate = (
            f"repository = {_sql_quote(repository)} "
            f"AND symbol_kind = {_sql_quote(symbol_kind)}"
        )
        table = tbl.to_lance().to_table(filter=predicate, columns=list(_PAYLOAD_FIELDS))
        chunks = [chunk_from_payload(row) for row in table.to_pylist()]
        return [c for c in chunks if c.symbol == symbol][:_SYMBOL_LOOKUP_LIMIT]


def _row_score(row: dict[str, Any]) -> float:
    """Relevance score of a result row.

    Hybrid rows carry the RRF-fused ``_relevance_score``; vector-only
    rows carry a cosine ``_distance`` (0 = identical), normalized into
    the same [0, 1] range.
    """
    score = row.get("_relevance_score")
    if score is not None:
        return float(score)
    distance = row.get("_distance")
    if distance is not None:
        return max(0.0, 1.0 - float(distance) / 2.0)
    return 0.0
