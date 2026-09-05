"""sqlite-vec-backed vector store — the only module that knows sqlite-vec.

Design notes
------------
- One embedded SQLite database file (``index.sqlite`` under ``data_dir``)
  via the stdlib ``sqlite3`` module plus the ``sqlite-vec`` extension.
  No separate server process is required.
- Three collections (``hdl``, ``docs``, ``code``) live in the same
  database, each as a triplet of tables:
    ``chunks_<c>``  — regular table: the chunk payload (id, attribution,
                      content, symbols as a JSON list, ...);
    ``vec_<c>``     — a sqlite-vec ``vec0`` table holding the dense
                      vectors, rowid-joined to ``chunks_<c>``;
    ``fts_<c>``     — a SQLite FTS5 table over the chunk content,
                      rowid-joined to ``chunks_<c>``.
  The database layout is versioned (``meta`` table; see
  :meth:`VectorStore.migrate`). Upgrading from the v1 layout (a single
  ``vhdl`` collection) is a deterministic full reindex: the migration
  drops the old tables safely because every chunk is reproducible from
  the git history.
- Chunk rows are keyed by a deterministic ``uuid5`` id over
  :meth:`Chunk.canonical_id` (unique column), so re-upserts are
  idempotent ``ON CONFLICT(id) DO UPDATE`` merges; stale-chunk cleanup
  happens via SQL predicates (repository/file), never by recomputing
  IDs.
- Search legs are selectable per query (``mode`` on
  :meth:`VectorStore.query`): hybrid (the default — the dense leg is a
  flat (exact) vec0 KNN over unit-normalized vectors, L2 order equals
  cosine order; the full-text leg is an FTS5 BM25 rank; the two rank
  lists are fused with RRF, rank-based, K=60, in this module),
  semantic (dense leg only, cosine-equivalent similarity score), or
  lexical (full-text leg only, ranked by BM25 with a rank-based
  display score). In hybrid, FTS5's ``unicode61`` tokenizer is
  configured with ``tokenchars '_'`` so HDL identifiers stay whole
  (``fifo_ctrl`` is one token; a query for it matches only documents
  containing it). English stop words are dropped from the FTS query so
  common words in a natural-language query do not pollute the leg.
  This full-text leg replaces the BM25 sparse-vector leg of the Qdrant
  layout — and with it the separate BM25 embedding model.
- Vector search is flat (exact) KNN: at repository scale (hundreds to
  tens of thousands of chunks) that is both fast and exact (sqlite-vec
  scores L2 distance with SIMD). If a deployment grows beyond that, a
  different vector engine can be swapped in without touching the rest
  of the store.
- The database runs in WAL mode so the periodic sync writer never
  blocks concurrent queries.
- The MCP layer only ever talks about repositories, files, symbols, and
  results — SQLite/sqlite-vec terminology does not leak out of this
  module.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sqlite_vec  # type: ignore[import-untyped]

from .config import AppConfig
from .models import INDEX_SCHEMA_VERSION, Chunk, CollectionName, ContentType

logger = logging.getLogger(__name__)

EXTENSION_SUPPORT_ERROR = (
    "the stdlib SQLite in this Python build lacks loadable-extension "
    "support (no Connection.enable_load_extension), so the sqlite-vec "
    "extension cannot be loaded. Use CPython 3.14 (uv: 'uv python "
    "install 3.14') or a system/homebrew Python; the self-check "
    "reports the same at startup."
)


COLLECTION_HDL = "hdl"
COLLECTION_DOCS = "docs"
COLLECTION_CODE = "code"
ALL_COLLECTIONS: tuple[CollectionName, ...] = (
    CollectionName.HDL,
    CollectionName.DOCS,
    CollectionName.CODE,
)

# RRF rank constant for fusing the per-collection dense and full-text
# legs (the standard value; originally taken from Qdrant's FusionQuery
# default).
_RRF_K = 60

#: SQLite's default compiled bound-parameter limit (SQLITE_MAX_VARIABLE_NUMBER)
#: is 999-ish depending on build; batched IN-lists and executemany chunking
#: stay comfortably under it.
_MAX_IN_PARAMS = 900

# English stop words dropped from FTS queries: with a pure OR-of-tokens
# query they would match nearly every document and blur the exact-match
# leg. HDL identifiers (rst_n, fifo_ctrl, ...) are never stop words.
_STOP_WORDS = frozenset(
    (
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "like",
        "made",
        "make",
        "many",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    )
)

_FTS_QUERY_TOKEN_RE = re.compile(r"[^0-9a-zA-Z_]+")
_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")

# Metadata columns (everything :meth:`Chunk.payload` stores).
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

# Namespace for deterministic row IDs (stable across commits and
# processes; makes re-upserts idempotent).
_ID_NAMESPACE = uuid.NAMESPACE_URL
#: Intentionally NOT renamed with the vhdl-rag-mcp -> corvidex-mcp rebrand:
#: this string only namespaces the UUIDv5 hash below, it is never displayed,
#: and changing it would make every existing chunk ID in an already-built
#: index look "new" on the next sync (forcing a full re-embed and risking
#: orphaned old-prefixed rows if reconciliation ever misses one).
_ID_PREFIX = "vhdl-rag-mcp::"


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot perform an operation."""


def point_id(chunk: Chunk) -> str:
    """Deterministic UUIDv5 row ID for a chunk (stable across commits)."""
    return str(uuid.uuid5(_ID_NAMESPACE, _ID_PREFIX + chunk.canonical_id))


def _col(field: str) -> str:
    """Double-quoted column identifier (``commit`` is a SQLite keyword)."""
    return '"' + field + '"'


def _check_filter_key(key: str) -> None:
    if key not in _FILTER_KEYS:
        raise VectorStoreError(f"unknown filter key: {key!r}")


def _where_clause(
    must: Mapping[str, str] | None,
    should: Mapping[str, Sequence[str]] | None,
) -> tuple[str, list[Any]] | None:
    """SQL predicate over the ``chunks`` table alias plus bound params.

    ``must`` maps columns to equality values; ``should`` maps a column
    to a list of values the row matches if it contains ANY of them
    (``json_each`` over the cross-referencing ``symbols`` list).
    """
    parts: list[str] = []
    params: list[Any] = []
    if must:
        for key, value in must.items():
            _check_filter_key(key)
            parts.append(f"c.{_col(key)} = ?")
            params.append(value)
    if should:
        for key, values in should.items():
            _check_filter_key(key)
            value_list = list(values)
            if not value_list:
                continue
            marks = ", ".join("?" * len(value_list))
            parts.append(
                f"EXISTS (SELECT 1 FROM json_each(c.{_col(key)}) "
                f"WHERE json_each.value IN ({marks}))"
            )
            params.extend(value_list)
    if not parts:
        return None
    return " AND ".join(parts), params


def _fts_query(text: str) -> str | None:
    """FTS5 query for the full-text leg, or None when it cannot match.

    The raw query is tokenized the way the index tokenizer sees it
    (split on anything but letters/digits/underscore) so the query is a
    safe OR of identifier tokens — no FTS5 metacharacters reach the
    engine. English stop words are dropped (see ``_STOP_WORDS``).
    """
    tokens = [
        token
        for token in _FTS_QUERY_TOKEN_RE.split(text or "")
        if token and token.lower() not in _STOP_WORDS
    ]
    return " OR ".join(tokens) if tokens else None


def _unit_vector(vector: Sequence[float]) -> np.ndarray:
    """Normalize to unit length (L2 order over unit vectors == cosine)."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


def _serialize_vector(vector: np.ndarray) -> bytes:
    """Pack a dense vector for ``vec0`` storage.

    Byte-identical to ``sqlite_vec.serialize_float32`` (little-endian
    float32), without that helper's Python-list round trip.
    """
    return np.asarray(vector, dtype="<f4").tobytes()


def _batched[T](seq: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield ``seq`` in chunks of at most ``size`` (keeps IN-lists and
    executemany batches under SQLite's bound-parameter limit)."""
    return (list(seq[i : i + size]) for i in range(0, len(seq), size))


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from its stored metadata row."""
    symbols_raw = payload.get("symbols")
    if isinstance(symbols_raw, str):
        symbols_raw = json.loads(symbols_raw)
    symbols_raw = symbols_raw or []
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
    """Thin, typed wrapper around an embedded SQLite + sqlite-vec index."""

    def __init__(self, config: AppConfig) -> None:
        path = config.sqlite_index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        # sqlite-vec is a loadable extension; the stdlib connection
        # refuses loadable extensions until explicitly enabled. Some
        # CPython builds link a SQLite without loadable-extension
        # support and lack the API entirely (e.g. some uv standalone
        # 3.12/3.13 builds): fail with an actionable error instead of
        # a bare AttributeError.
        if not hasattr(conn, "enable_load_extension"):
            raise VectorStoreError(EXTENSION_SUPPORT_ERROR)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        # WAL: the periodic sync writer never blocks concurrent queries.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        # Layout-version bookkeeping (see :meth:`migrate`).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()
        # Cache of collection names with a ``chunks_<name>`` table, refreshed
        # only by :meth:`_create_collection` and :meth:`_drop_legacy_vhdl`:
        # the table set otherwise never changes for the life of the process,
        # so hot read/write paths check this set instead of sqlite_master.
        self._known_collections: set[str] = self._tables()
        logger.info("sqlite-vec embedded store: %s", path)

    def close(self) -> None:
        self._conn.close()

    # -- tables ---------------------------------------------------------

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _tables(self) -> set[str]:
        """Collection names that have a ``chunks_<name>`` table."""
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {
            name[7:] for name in (row[0] for row in rows) if name.startswith("chunks_")
        }

    def _vec_dimension(self, collection: str) -> int | None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (f"vec_{collection}",),
        ).fetchone()
        if row is None:
            return None
        match = _VEC_DIM_RE.search(row[0])
        return int(match.group(1)) if match else None

    def _create_collection(self, collection: str, dim: int) -> None:
        cols = ", ".join(
            f'"{field}" '
            + ("INTEGER" if field in ("start_line", "end_line") else "TEXT")
            for field in _PAYLOAD_FIELDS
        )
        self._conn.execute(
            f"CREATE TABLE chunks_{collection} (id TEXT PRIMARY KEY, {cols})"
        )
        repo, file = _col("repository"), _col("file")
        # A single (repository, file) index: it also serves plain
        # repository-only lookups as its leftmost prefix, so a separate
        # single-column index would be redundant write amplification.
        # (symbol_kind has no equality-filter caller, so it gets no index.)
        self._conn.execute(
            f"CREATE INDEX idx_{collection}_repo_file "
            f"ON chunks_{collection}({repo}, {file})"
        )
        # The vec0 table's declared dimension is the drift guard (see
        # ensure_collections).
        self._conn.execute(
            f"CREATE VIRTUAL TABLE vec_{collection} USING vec0(embedding float[{dim}])"
        )
        # FTS5 over the content; tokenchars '_' keeps HDL identifiers
        # (fifo_ctrl, rst_n) single tokens so exact-identifier queries
        # match whole identifiers, not just their parts.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE fts_{collection} "
            "USING fts5(content, tokenize=\"unicode61 tokenchars '_'\")"
        )
        self._conn.commit()
        self._known_collections.add(collection)
        logger.info("created sqlite-vec collection %r (dim=%d)", collection, dim)

    def _drop_redundant_indexes(self, collection: str) -> None:
        """Drop indexes made redundant by a leftmost-prefix index or by a
        removed caller, on a database created before this cleanup.

        Idempotent (``IF EXISTS``): safe to run on every startup.
        """
        self._conn.execute(f"DROP INDEX IF EXISTS idx_{collection}_repo")
        self._conn.execute(f"DROP INDEX IF EXISTS idx_{collection}_symbol")
        self._conn.commit()

    def ensure_collections(self, hdl_dim: int, docs_dim: int, code_dim: int) -> None:
        """Create the collections if missing; fail loudly on dimension drift.

        A changed embedding model changes the dense vector size; silently
        mixing dimensions would corrupt retrieval, so this is an error
        with an actionable message. A dimension of 0 means the
        collection's embedding model is unavailable (degraded startup):
        the collection is skipped rather than created with an unknown
        size.
        """
        for name, dim in (
            (COLLECTION_HDL, hdl_dim),
            (COLLECTION_DOCS, docs_dim),
            (COLLECTION_CODE, code_dim),
        ):
            if dim == 0:
                continue
            if name not in self._known_collections:
                self._create_collection(name, dim)
                continue
            size = self._vec_dimension(name)
            if size != dim:
                raise VectorStoreError(
                    f"collection {name!r} has dense vector size {size}, "
                    f"expected {dim}. The embedding model changed — "
                    "delete the index (or the data directory) and reindex."
                )
            self._drop_redundant_indexes(name)

    # -- schema versioning ------------------------------------------------------

    _SCHEMA_VERSION_KEY = "schema_version"

    @property
    def schema_version(self) -> int:
        """The store layout version (``meta`` table, or inferred)."""
        version = self._read_version()
        return version if version is not None else self._infer_version()

    def _read_version(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (self._SCHEMA_VERSION_KEY,)
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _infer_version(self) -> int:
        """Infer the layout version from the tables present (no meta row).

        A database with no collection tables at all — fresh, or created
        while every embedding model was unavailable — is current-layout
        by definition: there is nothing to migrate.
        """
        tables = self._tables()
        if "vhdl" in tables:
            return 1
        return INDEX_SCHEMA_VERSION

    def _stamp_version(self, version: int) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (self._SCHEMA_VERSION_KEY, str(version)),
        )
        self._conn.commit()

    def migrate(self) -> bool:
        """Bring the store layout up to the current schema version.

        Migrations are small, deterministic steps applied in version
        order; the current version is tracked in the ``meta`` table and
        a database without a version row is inferred from its tables.
        Returns True when a data-changing migration ran (a current
        database is only re-stamped, never modified).
        """
        version = self._read_version()
        if version is None:
            version = self._infer_version()
        changed = False
        while version < INDEX_SCHEMA_VERSION:
            version += 1
            changed = self._apply_migration(version) or changed
        self._stamp_version(INDEX_SCHEMA_VERSION)
        if changed:
            logger.info("sqlite index migrated to schema v%d", INDEX_SCHEMA_VERSION)
        return changed

    def _apply_migration(self, target_version: int) -> bool:
        """Apply the single migration step landing on ``target_version``."""
        if target_version == 2:
            return self._drop_legacy_vhdl()
        raise VectorStoreError(f"unknown migration target: {target_version}")

    def _drop_legacy_vhdl(self) -> bool:
        """Drop the v1-layout ``vhdl`` collection (safe: reproducible)."""
        dropped = False
        for name in ("chunks_vhdl", "vec_vhdl", "fts_vhdl"):
            if self._table_exists(name):
                self._conn.execute(f"DROP TABLE {name}")
                dropped = True
        self._known_collections.discard("vhdl")
        if dropped:
            self._conn.commit()
            logger.info("deleted legacy sqlite-vec collection 'vhdl' (v1 layout)")
        return dropped

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
            self._upsert_collection(collection_name, items)

    def _payload_row(self, chunk: Chunk) -> tuple[Any, ...]:
        values: list[Any] = []
        for field in _PAYLOAD_FIELDS:
            if field == "symbols":
                values.append(
                    json.dumps(list(chunk.symbols)) if chunk.symbols else "[]"
                )
            elif field == "content_type":
                values.append(chunk.content_type.value)
            elif field == "collection":
                values.append(chunk.collection.value)
            else:
                values.append(getattr(chunk, field))
        return (point_id(chunk), *values)

    def _rowids_for_ids(self, collection: str, ids: Sequence[str]) -> dict[str, int]:
        """``{id: rowid}`` for the ids of ``collection`` that currently exist."""
        rowid_by_id: dict[str, int] = {}
        for batch in _batched(ids, _MAX_IN_PARAMS):
            marks = ", ".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT id, rowid FROM chunks_{collection} WHERE id IN ({marks})",
                batch,
            ).fetchall()
            rowid_by_id.update((row[0], row[1]) for row in rows)
        return rowid_by_id

    def _upsert_collection(
        self, collection: str, items: list[tuple[Chunk, list[float]]]
    ) -> None:
        conn = self._conn
        # Later duplicates of the same id win (matches the old sequential
        # delete-then-insert loop); collapsing them upfront also keeps the
        # batched vec/fts inserts below from targeting one rowid twice.
        by_id = {point_id(chunk): (chunk, d) for chunk, d in items}
        ids = list(by_id.keys())
        fields_sql = ", ".join(_col(f) for f in _PAYLOAD_FIELDS)
        placeholders = ", ".join("?" * len(_PAYLOAD_FIELDS))
        updates_sql = ", ".join(
            f"{_col(f)} = excluded.{_col(f)}" for f in _PAYLOAD_FIELDS
        )
        upsert_sql = (
            f"INSERT INTO chunks_{collection} (id, {fields_sql}) "
            f"VALUES (?, {placeholders}) ON CONFLICT(id) DO UPDATE SET {updates_sql}"
        )
        vec_sql = f"INSERT INTO vec_{collection} (rowid, embedding) VALUES (?, ?)"
        fts_sql = f"INSERT INTO fts_{collection} (rowid, content) VALUES (?, ?)"
        try:
            # Rows already present hit the ON CONFLICT UPDATE branch, which
            # keeps their rowid, so their stale vec0/fts5 rows (keyed by
            # that same rowid) must be cleared before the row is
            # re-inserted below. A fresh row gets a brand-new rowid that
            # was never used, so it needs no such cleanup.
            existing_rowids = self._rowids_for_ids(collection, ids)
            conn.executemany(
                upsert_sql, (self._payload_row(chunk) for chunk, _ in by_id.values())
            )
            for batch in _batched(list(existing_rowids.values()), _MAX_IN_PARAMS):
                marks = ", ".join("?" * len(batch))
                conn.execute(
                    f"DELETE FROM vec_{collection} WHERE rowid IN ({marks})", batch
                )
                conn.execute(
                    f"DELETE FROM fts_{collection} WHERE rowid IN ({marks})", batch
                )
            rowid_by_id = self._rowids_for_ids(collection, ids)
            conn.executemany(
                vec_sql,
                (
                    (rowid_by_id[chunk_id], _serialize_vector(_unit_vector(d)))
                    for chunk_id, (_, d) in by_id.items()
                ),
            )
            conn.executemany(
                fts_sql,
                (
                    (rowid_by_id[chunk_id], chunk.content)
                    for chunk_id, (chunk, _) in by_id.items()
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"upsert into collection {collection!r} failed: {exc}"
            ) from exc
        logger.info("upserted %d chunks into collection %r", len(by_id), collection)

    def _delete_where(
        self, name: str, predicate_sql: str, params: Sequence[Any], what: str
    ) -> int:
        """Delete rows of one collection matching a WHERE predicate.

        Shared by :meth:`delete_repository`, :meth:`delete_file_prefix` and
        :meth:`delete_files`; the caller owns the transaction (commit or
        rollback), so this never commits.
        """
        if name not in self._known_collections:
            return 0
        conn = self._conn
        rowids = [
            row[0]
            for row in conn.execute(
                f"SELECT rowid FROM chunks_{name} WHERE {predicate_sql}", params
            )
        ]
        if not rowids:
            return 0
        for batch in _batched(rowids, _MAX_IN_PARAMS):
            marks = ", ".join("?" * len(batch))
            conn.execute(f"DELETE FROM vec_{name} WHERE rowid IN ({marks})", batch)
            conn.execute(f"DELETE FROM fts_{name} WHERE rowid IN ({marks})", batch)
        conn.execute(f"DELETE FROM chunks_{name} WHERE {predicate_sql}", params)
        logger.info("deleted %d stale chunks from %r for %s", len(rowids), name, what)
        return len(rowids)

    def delete_file(self, repository: str, file: str) -> int:
        """Remove all chunks for one file from every collection."""
        return self.delete_files(repository, [file])

    def delete_files(self, repository: str, files: Iterable[str]) -> int:
        """Remove all chunks for a batch of files from every collection.

        A single transaction covers every collection and file batch
        (``file IN (...)``, chunked to stay under SQLite's bound-parameter
        limit) — the caller no longer pays one transaction per file.
        """
        file_list = list(files)
        if not file_list:
            return 0
        conn = self._conn
        total = 0
        try:
            for collection in ALL_COLLECTIONS:
                name = collection.value
                for batch in _batched(file_list, _MAX_IN_PARAMS - 1):
                    marks = ", ".join("?" * len(batch))
                    predicate = (
                        f"{_col('repository')} = ? AND {_col('file')} IN ({marks})"
                    )
                    total += self._delete_where(
                        name,
                        predicate,
                        [repository, *batch],
                        f"repository={repository} files={len(batch)}",
                    )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"delete_files from repository {repository!r} failed: {exc}"
            ) from exc
        return total

    def delete_repository(self, repository: str) -> int:
        conn = self._conn
        total = 0
        try:
            for collection in ALL_COLLECTIONS:
                total += self._delete_where(
                    collection.value,
                    f"{_col('repository')} = ?",
                    [repository],
                    f"repository={repository}",
                )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"delete from repository {repository!r} failed: {exc}"
            ) from exc
        return total

    def delete_file_prefix(self, repository: str, prefix: str) -> int:
        """Remove every chunk whose file path lies under ``prefix/``
        (submodule prefix purge)."""
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"{escaped}/%"
        predicate = f"{_col('repository')} = ? AND {_col('file')} LIKE ? ESCAPE '\\'"
        params = [repository, pattern]
        conn = self._conn
        total = 0
        try:
            for collection in ALL_COLLECTIONS:
                total += self._delete_where(
                    collection.value,
                    predicate,
                    params,
                    f"repository={repository} prefix={prefix}/",
                )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"delete from repository {repository!r} failed: {exc}"
            ) from exc
        return total

    # -- reads ----------------------------------------------------------

    def query(
        self,
        collection: CollectionName,
        dense: list[float],
        query_text: str,
        limit: int,
        must: Mapping[str, str] | None = None,
        should: Mapping[str, Sequence[str]] | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredChunk]:
        """Search one collection with row filters.

        ``mode`` selects the legs:

        * ``"hybrid"`` (default) — dense + full-text, rank lists fused
          with RRF; an empty full-text leg falls back to the dense leg
          alone.
        * ``"semantic"`` — the dense leg only; the score is the
          cosine-equivalent similarity in [0, 1].
        * ``"lexical"`` — the full-text leg only, ranked by BM25; the
          score is the RRF term of the row's rank (same scale family
          as hybrid, so scores stay comparable across modes).
        """
        if mode not in ("hybrid", "semantic", "lexical"):
            raise VectorStoreError(f"unknown query mode {mode!r}")
        name = collection.value
        if name not in self._known_collections:
            return []
        where = _where_clause(must, should)
        where_sql = f" AND {where[0]}" if where else ""
        where_params = where[1] if where else []
        conn = self._conn

        dense_rows: list[Any] = []
        if mode in ("hybrid", "semantic"):
            vector = _unit_vector(dense)
            dense_sql = (
                f"SELECT v.rowid, v.distance FROM vec_{name} v "
                f"JOIN chunks_{name} c ON c.rowid = v.rowid "
                f"WHERE v.embedding MATCH ? AND v.k = ?{where_sql} "
                "ORDER BY v.distance"
            )
            dense_rows = conn.execute(
                dense_sql,
                (_serialize_vector(vector), limit, *where_params),
            ).fetchall()

        fts_query = _fts_query(query_text or "")
        fts_rows: list[Any] = []
        if mode in ("hybrid", "lexical") and fts_query is not None:
            fts_sql = (
                f"SELECT f.rowid, bm25(fts_{name}) AS score FROM fts_{name} f "
                f"JOIN chunks_{name} c ON c.rowid = f.rowid "
                f"WHERE fts_{name} MATCH ?{where_sql} "
                "ORDER BY score LIMIT ?"
            )
            fts_rows = conn.execute(
                fts_sql, (fts_query, *where_params, limit)
            ).fetchall()

        if mode == "lexical":
            # BM25-ranked full text only. BM25 itself is an unbounded
            # engine score, so the display score is the RRF term of the
            # row's rank — best first, same scale family as hybrid.
            results = [
                (row[0], 1.0 / (_RRF_K + rank))
                for rank, row in enumerate(fts_rows, start=1)
            ]
        elif fts_query is None or mode == "semantic":
            # Dense-only: map the (cosine-equivalent) distance into the
            # same [0, 1] range the hybrid score uses.
            results = []
            for rowid, distance in dense_rows:
                distance = float(distance)
                cosine_distance = distance * distance / 2.0  # unit vectors
                results.append((rowid, max(0.0, 1.0 - cosine_distance / 2.0)))
        else:
            scores: dict[int, float] = {}
            for rank, row in enumerate(dense_rows, start=1):
                scores[row[0]] = scores.get(row[0], 0.0) + 1.0 / (_RRF_K + rank)
            for rank, row in enumerate(fts_rows, start=1):
                scores[row[0]] = scores.get(row[0], 0.0) + 1.0 / (_RRF_K + rank)
            results = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        top = results[:limit]
        if not top:
            return []
        marks = ", ".join("?" * len(top))
        rows = conn.execute(
            f"SELECT rowid, * FROM chunks_{name} WHERE rowid IN ({marks})",
            [rowid for rowid, _ in top],
        ).fetchall()
        by_rowid = {row["rowid"]: row for row in rows}
        return [
            ScoredChunk(score=score, chunk=chunk_from_payload(dict(by_rowid[rowid])))
            for rowid, score in top
            if rowid in by_rowid
        ]

    def count(self, collection: CollectionName | None = None) -> int:
        if collection is not None:
            return self._count(collection.value)
        return sum(self._count(c.value) for c in ALL_COLLECTIONS)

    def _count(self, collection: str) -> int:
        if collection not in self._known_collections:
            return 0
        row = self._conn.execute(f"SELECT count(*) FROM chunks_{collection}").fetchone()
        return int(row[0])

    def count_repository(
        self, repository: str, collection: CollectionName | None = None
    ) -> int:
        """Rows of one repository, in one collection or all of them."""
        if collection is not None:
            if collection.value not in self._known_collections:
                return 0
            row = self._conn.execute(
                f"SELECT count(*) FROM chunks_{collection.value} WHERE repository = ?",
                (repository,),
            ).fetchone()
            return int(row[0])
        return sum(self.count_repository(repository, c) for c in ALL_COLLECTIONS)

    def list_files(
        self, repository: str, collection: CollectionName | None = None
    ) -> list[str]:
        """Distinct indexed file paths of one repository (one collection or
        all of them), sorted. A file is listed when at least one of its
        chunks is indexed — exactly the set ``get_source`` can read."""
        files: set[str] = set()
        for c in [collection] if collection is not None else list(ALL_COLLECTIONS):
            if c.value not in self._known_collections:
                continue
            name = f"chunks_{c.value}"
            rows = self._conn.execute(
                f"SELECT DISTINCT {_col('file')} FROM {name} WHERE repository = ?",
                (repository,),
            ).fetchall()
            files.update(row[0] for row in rows)
        return sorted(files)

    def chunks_for_file(self, repository: str, file: str) -> list[Chunk]:
        """All currently indexed chunks for one file (every collection)."""
        chunks: list[Chunk] = []
        for collection in ALL_COLLECTIONS:
            name = collection.value
            if name not in self._known_collections:
                continue
            rows = self._conn.execute(
                f"SELECT * FROM chunks_{name} "
                f"WHERE {_col('repository')} = ? AND {_col('file')} = ?",
                (repository, file),
            ).fetchall()
            chunks.extend(chunk_from_payload(dict(row)) for row in rows)
        return chunks
