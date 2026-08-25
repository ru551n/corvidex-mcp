"""Qdrant-backed vector store — the only module that knows Qdrant.

Design notes
------------
- One embedded (local) Qdrant instance under ``<data_dir>/qdrant`` by
  default; server mode (``[qdrant] mode = "server"``) is supported via the
  same interface. No separate Qdrant server is required and no second
  instance is run.
- Three collections in that single instance:
    ``hdl``   — semantically chunked HDL (VHDL, Verilog, SystemVerilog)
    ``docs``  — HDL-related documentation
    ``code``  — general source code (C/C++, Python, ...)
  Collections may use different dense embedding models; the store keeps
  them isolated. Upgrading from the v1 layout (a ``vhdl`` collection)
  is a deterministic full reindex; :meth:`VectorStore.delete_legacy_vhdl`
  drops the old collection safely because every chunk is reproducible
  from the git history.
- Every collection carries two named vectors:
    ``dense``  — dense vector from the domain's embedding model
    ``sparse`` — BM25 sparse vector (Qdrant-native sparse index)
  Search is a native Qdrant hybrid query: both legs are prefetched and
  fused with RRF, so results combine semantic similarity and exact
  identifier/token matching in one ranked list.
- Local mode only accepts UUID point IDs, so chunk IDs are mapped to
  deterministic ``uuid5`` values over :meth:`Chunk.canonical_id`.
  Determinism makes re-upserts idempotent; stale-chunk cleanup happens via
  payload filters (repository/file), never by recomputing IDs.
- The MCP layer only ever talks about repositories, files, symbols, and
  results — Qdrant terminology does not leak out of this module.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .config import AppConfig
from .models import Chunk, CollectionName, SparseVectorData

logger = logging.getLogger(__name__)

COLLECTION_HDL = "hdl"
COLLECTION_DOCS = "docs"
COLLECTION_CODE = "code"
ALL_COLLECTIONS: tuple[CollectionName, ...] = (
    CollectionName.HDL,
    CollectionName.DOCS,
    CollectionName.CODE,
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Namespace for deterministic point IDs (local Qdrant requires UUIDs).
_ID_NAMESPACE = uuid.NAMESPACE_URL
_ID_PREFIX = "vhdl-rag-mcp::"

# Payload fields that may be used as equality filters in queries.
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


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot perform an operation."""


def point_id(chunk: Chunk) -> str:
    """Deterministic UUIDv5 point ID for a chunk (local mode requires UUIDs)."""
    return str(uuid.uuid5(_ID_NAMESPACE, _ID_PREFIX + chunk.canonical_id))


def _check_filter_key(key: str) -> None:
    if key not in _FILTER_KEYS:
        raise VectorStoreError(f"unknown filter key: {key!r}")


def _build_filter(
    must: Mapping[str, str] | None,
    should: Mapping[str, Sequence[str]] | None,
) -> Filter | None:
    """AND of ``must`` equalities plus an optional OR of ``should`` values.

    ``should`` maps a payload key to a list of values: a point matches if
    the key equals ANY of them (this is how the cross-referencing ``symbols``
    list field is queried).
    """
    must_conditions: list[Condition] = []
    should_conditions: list[Condition] = []
    if must:
        for key, value in must.items():
            _check_filter_key(key)
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
    if should:
        for key, values in should.items():
            _check_filter_key(key)
            for value in values:
                should_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
    if not must_conditions and not should_conditions:
        return None
    return Filter(must=must_conditions or None, should=should_conditions or None)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its hybrid (RRF-fused) relevance score."""

    score: float
    chunk: Chunk


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from its stored payload."""
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


def _sparse_qdrant(data: SparseVectorData) -> SparseVector:
    return SparseVector(indices=list(data.indices), values=list(data.values))


class VectorStore:
    """Thin, typed wrapper around a single (local or server) Qdrant."""

    def __init__(self, config: AppConfig) -> None:
        if config.qdrant.mode == "local":
            path = config.qdrant_local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
            logger.info("qdrant local mode: %s", path)
        elif config.qdrant.url:
            self._client = QdrantClient(url=config.qdrant.url)
            logger.info("qdrant server mode: %s", config.qdrant.url)
        else:
            raise VectorStoreError(
                'qdrant.mode = "server" requires qdrant.url to be configured'
            )
        self._existing: set[str] | None = None

    def close(self) -> None:
        self._client.close()

    # -- collections ----------------------------------------------------

    def _collections(self) -> set[str]:
        if self._existing is None:
            self._existing = {
                c.name for c in self._client.get_collections().collections
            }
        return self._existing

    def _dense_dimension(self, name: str) -> int | None:
        vectors = self._client.get_collection(name).config.params.vectors
        if isinstance(vectors, VectorParams):
            return vectors.size
        if isinstance(vectors, dict):
            dense = vectors.get(DENSE_VECTOR_NAME)
            return dense.size if isinstance(dense, VectorParams) else None
        return None

    def ensure_collections(self, hdl_dim: int, docs_dim: int, code_dim: int) -> None:
        """Create the collections if missing; fail loudly on dimension drift.

        A changed embedding model changes the dense vector size; silently
        mixing dimensions would corrupt retrieval, so this is an error with
        an actionable message.
        """
        for name, dim in (
            (COLLECTION_HDL, hdl_dim),
            (COLLECTION_DOCS, docs_dim),
            (COLLECTION_CODE, code_dim),
        ):
            if name in self._collections():
                size = self._dense_dimension(name)
                if size != dim:
                    raise VectorStoreError(
                        f"collection {name!r} has dense vector size {size}, "
                        f"expected {dim}. The embedding model changed — "
                        "delete the collection (or the qdrant data directory) "
                        "and reindex."
                    )
                continue
            self._client.create_collection(
                name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())
                },
            )
            if self._existing is not None:
                self._existing.add(name)
            logger.info("created qdrant collection %r (dim=%d)", name, dim)

    def delete_legacy_vhdl(self) -> bool:
        """Drop the v1-layout ``vhdl`` collection (safe: reproducible).

        Returns True when a legacy collection was removed.
        """
        name = "vhdl"
        if name not in self._collections():
            return False
        self._client.delete_collection(name)
        if self._existing is not None:
            self._existing.discard(name)
        logger.info("deleted legacy qdrant collection %r (v1 layout)", name)
        return True

    # -- writes ---------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVectorData],
    ) -> None:
        """Upsert chunks with precomputed vectors (same order, same length)."""
        if len(chunks) != len(dense) or len(chunks) != len(sparse):
            raise VectorStoreError(
                f"got {len(chunks)} chunks, {len(dense)} dense vectors, "
                f"{len(sparse)} sparse vectors"
            )
        by_collection: dict[str, list[tuple[Chunk, list[float], SparseVectorData]]] = {}
        for chunk, d, s in zip(chunks, dense, sparse, strict=True):
            by_collection.setdefault(chunk.collection.value, []).append((chunk, d, s))
        for collection_name, items in by_collection.items():
            self._client.upsert(
                collection_name,
                points=[
                    PointStruct(
                        id=point_id(chunk),
                        vector={
                            DENSE_VECTOR_NAME: d,
                            SPARSE_VECTOR_NAME: _sparse_qdrant(s),
                        },
                        payload=chunk.payload(),
                    )
                    for chunk, d, s in items
                ],
                wait=True,
            )
            logger.info(
                "upserted %d chunks into collection %r", len(items), collection_name
            )

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
        if name not in self._collections():
            return 0
        before = self._client.count(name, exact=True).count
        conditions: list[Condition] = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in must.items()
        ]
        self._client.delete(name, points_selector=Filter(must=conditions), wait=True)
        after = self._client.count(name, exact=True).count
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
        sparse: SparseVectorData,
        limit: int,
        must: Mapping[str, str] | None = None,
        should: Mapping[str, Sequence[str]] | None = None,
    ) -> list[ScoredChunk]:
        """Hybrid (dense + sparse, RRF-fused) search with payload filters.

        The filter is applied to every prefetch leg: for fusion queries
        Qdrant does not apply a top-level ``query_filter`` to the fused
        results, so filtering must happen on each leg.
        """
        name = collection.value
        if name not in self._collections():
            return []
        query_filter = _build_filter(must, should)
        prefetch = [
            Prefetch(
                query=list(dense),
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
            )
        ]
        if not sparse.is_empty:
            prefetch.append(
                Prefetch(
                    query=_sparse_qdrant(sparse),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                )
            )
        response = self._client.query_points(
            name,
            query=FusionQuery(fusion=Fusion.RRF),
            prefetch=prefetch,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            ScoredChunk(
                score=point.score, chunk=chunk_from_payload(point.payload or {})
            )
            for point in response.points
            if point.payload is not None
        ]

    def count(self, collection: CollectionName | None = None) -> int:
        if collection is not None:
            if collection.value not in self._collections():
                return 0
            return self._client.count(collection.value, exact=True).count
        return sum(self.count(c) for c in ALL_COLLECTIONS)

    def chunks_for_file(self, repository: str, file: str) -> list[Chunk]:
        """All currently indexed chunks for one file (every collection)."""
        chunks: list[Chunk] = []
        for collection in ALL_COLLECTIONS:
            name = collection.value
            if name not in self._collections():
                continue
            records, _ = self._client.scroll(
                name,
                limit=1024,
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repository", match=MatchValue(value=repository)
                        ),
                        FieldCondition(key="file", match=MatchValue(value=file)),
                    ]
                ),
            )
            chunks.extend(
                chunk_from_payload(r.payload or {})
                for r in records
                if r.payload is not None
            )
        return chunks

    def get_by_symbol(
        self, repository: str, symbol: str, symbol_kind: str
    ) -> list[Chunk]:
        """HDL chunks of one kind named ``symbol`` in ``repository``."""
        name = COLLECTION_HDL
        if name not in self._collections():
            return []
        records, _ = self._client.scroll(
            name,
            limit=64,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="repository", match=MatchValue(value=repository)
                    ),
                    FieldCondition(
                        key="symbol_kind", match=MatchValue(value=symbol_kind)
                    ),
                ]
            ),
        )
        return [
            chunk_from_payload(r.payload or {})
            for r in records
            if r.payload is not None and r.payload.get("symbol") == symbol
        ]
