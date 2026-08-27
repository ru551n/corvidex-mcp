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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sqlite_vec  # type: ignore[import-untyped]

from .config import AppConfig
from .models import INDEX_SCHEMA_VERSION, Chunk, CollectionName

logger = logging.getLogger(__name__)

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

#: Rows returned by get_by_symbol are capped (a symbol rarely appears in
#: more than a few constructs of one kind).
_SYMBOL_LOOKUP_LIMIT = 64

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
            ors: list[str] = []
            for value in values:
                ors.append(
                    f"EXISTS (SELECT 1 FROM json_each(c.{_col(key)}) "
                    "WHERE json_each.value = ?)"
                )
                params.append(value)
            if ors:
                parts.append("(" + " OR ".join(ors) + ")")
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


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from its stored metadata row."""
    from .models import CollectionName, ContentType

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
        # refuses loadable extensions until explicitly enabled.
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
        repo, file, kind = (
            _col("repository"),
            _col("file"),
            _col("symbol_kind"),
        )
        self._conn.execute(
            f"CREATE INDEX idx_{collection}_repo ON chunks_{collection}({repo})"
        )
        self._conn.execute(
            f"CREATE INDEX idx_{collection}_repo_file "
            f"ON chunks_{collection}({repo}, {file})"
        )
        self._conn.execute(
            f"CREATE INDEX idx_{collection}_symbol "
            f"ON chunks_{collection}({kind}, {repo})"
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
        logger.info("created sqlite-vec collection %r (dim=%d)", collection, dim)

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
            if not self._table_exists(f"chunks_{name}"):
                self._create_collection(name, dim)
                continue
            size = self._vec_dimension(name)
            if size != dim:
                raise VectorStoreError(
                    f"collection {name!r} has dense vector size {size}, "
                    f"expected {dim}. The embedding model changed — "
                    "delete the index (or the data directory) and reindex."
                )

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
            if not items:
                continue
            self._upsert_collection(collection_name, items)

    def _payload_row(self, chunk: Chunk) -> tuple[Any, ...]:
        payload = chunk.payload()
        values: list[Any] = []
        for field in _PAYLOAD_FIELDS:
            value = payload.get(field)
            if field == "symbols":
                value = json.dumps(list(value)) if value else "[]"
            values.append(value)
        return (point_id(chunk), *values)

    def _upsert_collection(
        self, collection: str, items: list[tuple[Chunk, list[float]]]
    ) -> None:
        conn = self._conn
        fields_sql = ", ".join(_col(f) for f in _PAYLOAD_FIELDS)
        placeholders = ", ".join("?" * len(_PAYLOAD_FIELDS))
        updates_sql = ", ".join(
            f"{_col(f)} = excluded.{_col(f)}" for f in _PAYLOAD_FIELDS
        )
        upsert_sql = (
            f"INSERT INTO chunks_{collection} (id, {fields_sql}) "
            f"VALUES (?, {placeholders}) ON CONFLICT(id) DO UPDATE SET "
            f"{updates_sql} RETURNING rowid"
        )
        vec_sql = f"INSERT INTO vec_{collection} (rowid, embedding) VALUES (?, ?)"
        fts_sql = f"INSERT INTO fts_{collection} (rowid, content) VALUES (?, ?)"
        try:
            for chunk, d in items:
                rowid = conn.execute(upsert_sql, self._payload_row(chunk)).fetchone()[0]
                vector = _unit_vector(d)
                # vec0/fts5 are virtual tables: no OR REPLACE, so an
                # upsert is delete-then-insert (same transaction).
                conn.execute(f"DELETE FROM vec_{collection} WHERE rowid = ?", (rowid,))
                conn.execute(
                    vec_sql, (rowid, sqlite_vec.serialize_float32(vector.tolist()))
                )
                conn.execute(f"DELETE FROM fts_{collection} WHERE rowid = ?", (rowid,))
                conn.execute(fts_sql, (rowid, chunk.content))
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"upsert into collection {collection!r} failed: {exc}"
            ) from exc
        logger.info("upserted %d chunks into collection %r", len(items), collection)

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
        if not self._table_exists(f"chunks_{name}"):
            return 0
        predicate = " AND ".join(f"{_col(key)} = ?" for key in must)
        params = [must[key] for key in must]
        conn = self._conn
        try:
            rowids = [
                row[0]
                for row in conn.execute(
                    f"SELECT rowid FROM chunks_{name} WHERE {predicate}", params
                )
            ]
            if not rowids:
                return 0
            conn.executemany(
                f"DELETE FROM vec_{name} WHERE rowid = ?", [(r,) for r in rowids]
            )
            conn.executemany(
                f"DELETE FROM fts_{name} WHERE rowid = ?", [(r,) for r in rowids]
            )
            conn.execute(f"DELETE FROM chunks_{name} WHERE {predicate}", params)
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise VectorStoreError(
                f"delete from collection {name!r} failed: {exc}"
            ) from exc
        if rowids:
            logger.info(
                "deleted %d stale chunks from %r for %s",
                len(rowids),
                name,
                " ".join(f"{k}={v}" for k, v in must.items()),
            )
        return len(rowids)

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
        if not self._table_exists(f"chunks_{name}"):
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
                (sqlite_vec.serialize_float32(vector.tolist()), limit, *where_params),
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
        if not self._table_exists(f"chunks_{collection}"):
            return 0
        row = self._conn.execute(f"SELECT count(*) FROM chunks_{collection}").fetchone()
        return int(row[0])

    def count_repository(
        self, repository: str, collection: CollectionName | None = None
    ) -> int:
        """Rows of one repository, in one collection or all of them."""
        if collection is not None:
            if not self._table_exists(f"chunks_{collection.value}"):
                return 0
            row = self._conn.execute(
                f"SELECT count(*) FROM chunks_{collection.value} WHERE repository = ?",
                (repository,),
            ).fetchone()
            return int(row[0])
        return sum(self.count_repository(repository, c) for c in ALL_COLLECTIONS)

    def chunks_for_file(self, repository: str, file: str) -> list[Chunk]:
        """All currently indexed chunks for one file (every collection)."""
        chunks: list[Chunk] = []
        for collection in ALL_COLLECTIONS:
            name = collection.value
            if not self._table_exists(f"chunks_{name}"):
                continue
            rows = self._conn.execute(
                f"SELECT * FROM chunks_{name} "
                f"WHERE {_col('repository')} = ? AND {_col('file')} = ?",
                (repository, file),
            ).fetchall()
            chunks.extend(chunk_from_payload(dict(row)) for row in rows)
        return chunks

    def get_by_symbol(
        self, repository: str, symbol: str, symbol_kind: str
    ) -> list[Chunk]:
        """HDL chunks of one kind named ``symbol`` in ``repository``."""
        name = COLLECTION_HDL
        if not self._table_exists(f"chunks_{name}"):
            return []
        rows = self._conn.execute(
            f"SELECT * FROM chunks_{name} "
            f"WHERE {_col('repository')} = ? AND {_col('symbol_kind')} = ?",
            (repository, symbol_kind),
        ).fetchall()
        chunks = [chunk_from_payload(dict(row)) for row in rows]
        return [c for c in chunks if c.symbol == symbol][:_SYMBOL_LOOKUP_LIMIT]
