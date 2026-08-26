"""Embedding layer: local dense + sparse (BM25) embeddings via FastEmbed.

Hides the concrete implementation (FastEmbed ONNX models, no server)
behind a small interface. The vector store and retrieval only deal in
plain floats and :class:`~vhdl_rag_mcp.models.SparseVectorData`.

Two model families are used per collection:

- dense  ``jinaai/jina-embeddings-v2-*`` — semantic similarity
- sparse ``Qdrant/bm25`` — exact token matching (identifier search),
  fused with the dense leg by Qdrant's native hybrid (RRF) query.

Queries and indexed passages use different entry points: jina v2 models
require task prefixes ("query:"/"passage:"), which FastEmbed applies
through ``query_embed``/``passage_embed``; ``Qdrant/bm25`` likewise takes
a ``mode`` argument.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..models import SparseVectorData

logger = logging.getLogger(__name__)


@runtime_checkable
class DenseModelLike(Protocol):
    """The FastEmbed dense-model subset this provider relies on."""

    @property
    def embedding_size(self) -> int: ...

    def passage_embed(self, texts: list[str], **kwargs: Any) -> Iterable[object]: ...

    def query_embed(self, query: str, **kwargs: Any) -> Iterable[object]: ...


@runtime_checkable
class SparseVectorResult(Protocol):
    """Structural type for FastEmbed's sparse output (index/value arrays)."""

    indices: np.ndarray
    values: np.ndarray


@runtime_checkable
class SparseModelLike(Protocol):
    """The FastEmbed sparse-model subset this provider relies on."""

    def passage_embed(
        self, texts: list[str], **kwargs: Any
    ) -> Iterable[SparseVectorResult]: ...

    def query_embed(
        self, query: str, **kwargs: Any
    ) -> Iterable[SparseVectorResult]: ...


class EmbeddingProvider(Protocol):
    """Embeds batches of texts, dense and sparse."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_sparse_passages(self, texts: list[str]) -> list[SparseVectorData]: ...

    def embed_sparse_query(self, text: str) -> SparseVectorData: ...


def _sparse_from(result: SparseVectorResult) -> SparseVectorData:
    indices = np.asarray(result.indices, dtype=np.int32).ravel()
    values = np.asarray(result.values, dtype=np.float32).ravel()
    return SparseVectorData(
        indices=tuple(int(i) for i in indices),
        values=tuple(float(v) for v in values),
    )


class FastEmbedProvider:
    """FastEmbed-based provider (dense + sparse); models download on first use.

    Dense inference is memory-bounded: ONNX Runtime's per-thread memory
    arenas retain peak tensor sizes and never release them, and
    transformer attention work is quadratic in sequence length. With the
    model's full context (8192 tokens for the jina v2 models) and a
    32-thread pool, a single long passage in a batch can make the arena
    reserve tens of GB. ``max_tokens`` re-bounds the tokenizer truncation
    (fastembed does not expose it) and ``threads``/``batch_size`` keep
    the per-call peak small, so embedding stays bounded regardless of
    chunk length. The ONNX CPU memory arena can additionally be disabled
    (``enable_arena=False``, the default) to release peak buffers after
    each inference — lower resident RAM at a ~35% indexing-time cost.
    """

    def __init__(
        self,
        dense_model: str,
        sparse_model: str = "Qdrant/bm25",
        cache_dir: Path | None = None,
        dense: DenseModelLike | None = None,
        sparse: SparseModelLike | None = None,
        batch_size: int = 8,
        max_tokens: int | None = None,
        threads: int | None = None,
        enable_arena: bool = False,
    ) -> None:
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._threads = threads
        self._enable_arena = enable_arena
        self._dense: DenseModelLike | None = dense
        self._sparse: SparseModelLike | None = sparse
        self._cache_dir = cache_dir

    def _ensure_dense(self) -> None:
        """Lazily load the dense model (and only the dense model)."""
        if self._dense is None:
            from fastembed import TextEmbedding

            cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
            logger.info("loading dense model %s", self._dense_model)
            self._dense = TextEmbedding(
                self._dense_model,
                cache_dir=cache_dir,
                threads=self._threads,
                enable_cpu_mem_arena=self._enable_arena,
            )
            self._apply_token_cap(self._dense)
            logger.debug(
                "dense model %s: max_tokens=%s threads=%s batch_size=%d "
                "cpu_mem_arena=%s",
                self._dense_model,
                self._max_tokens,
                self._threads,
                self._batch_size,
                self._enable_arena,
            )

    def _apply_token_cap(self, dense: DenseModelLike) -> None:
        """Bound fastembed's tokenizer truncation to ``max_tokens``.

        FastEmbed truncates at the model's full context (8192 tokens for
        the jina v2 models) and does not expose a shorter limit; its
        tokenizer object is re-configurable. Inert for test fakes that do
        not carry the fastembed model/tokenizer attributes.
        """
        if self._max_tokens is None:
            return
        model = getattr(dense, "model", None)
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "enable_truncation"):
            tokenizer.enable_truncation(max_length=self._max_tokens)

    def _ensure_sparse(self) -> None:
        """Lazily load the sparse model (and only the sparse model)."""
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
            logger.info("loading sparse model %s", self._sparse_model)
            self._sparse = SparseTextEmbedding(self._sparse_model, cache_dir=cache_dir)

    @property
    def model_name(self) -> str:
        return self._dense_model

    @property
    def dimension(self) -> int:
        self._ensure_dense()
        assert self._dense is not None
        return int(self._dense.embedding_size)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_dense()
        assert self._dense is not None
        vectors = [
            np.asarray(v, dtype=np.float32).ravel()
            for v in self._dense.passage_embed(texts, batch_size=self._batch_size)
        ]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding model returned {len(vectors)} vectors "
                f"for {len(texts)} texts"
            )
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_dense()
        assert self._dense is not None
        vectors = [
            np.asarray(v, dtype=np.float32).ravel()
            for v in self._dense.query_embed(text, batch_size=self._batch_size)
        ]
        if not vectors:
            raise RuntimeError("embedding model returned no query vector")
        return [float(x) for x in vectors[0]]

    def embed_sparse_passages(self, texts: list[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        self._ensure_sparse()
        assert self._sparse is not None
        results = [
            _sparse_from(r)
            for r in self._sparse.passage_embed(list(texts), mode="passage")
        ]
        if len(results) != len(texts):
            raise RuntimeError(
                f"sparse model returned {len(results)} vectors for {len(texts)} texts"
            )
        return results

    def embed_sparse_query(self, text: str) -> SparseVectorData:
        self._ensure_sparse()
        assert self._sparse is not None
        results = list(self._sparse.query_embed(text, mode="query"))
        if not results:
            return SparseVectorData(indices=(), values=())
        return _sparse_from(results[0])
