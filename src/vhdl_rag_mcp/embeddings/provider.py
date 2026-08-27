"""Embedding layer: local dense embeddings via FastEmbed.

Hides the concrete implementation (FastEmbed ONNX models, no server)
behind a small interface. The vector store and retrieval only deal in
plain floats.

Queries and indexed passages use different entry points: jina v2 models
require task prefixes ("query:"/"passage:"), which FastEmbed applies
through ``query_embed``/``passage_embed``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class DenseModelLike(Protocol):
    """The FastEmbed dense-model subset this provider relies on."""

    @property
    def embedding_size(self) -> int: ...

    def passage_embed(self, texts: list[str], **kwargs: Any) -> Iterable[object]: ...

    def query_embed(self, query: str, **kwargs: Any) -> Iterable[object]: ...


class EmbeddingProvider(Protocol):
    """Embeds batches of texts."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _even_slices(count: int, workers: int) -> list[tuple[int, int]]:
    """Split ``count`` items into ``workers`` contiguous, balanced
    ``(start, end)`` half-open ranges. The largest slice holds at most
    one item more than the smallest (low per-slice padding variance)."""
    n = min(workers, count)
    if n <= 0:
        return []
    size = (count + n - 1) // n
    return [(i, min(i + size, count)) for i in range(0, count, size)]


def _embed_slice(
    args: tuple[
        str, str | None, str | None, int | None, bool, int | None, int, list[str]
    ],
) -> list[np.ndarray]:
    """Worker for data-parallel dense embedding (#3).

    Runs in a spawned child process with its own ONNX Runtime session.
    Receives already length-sorted, pre-truncated texts and returns
    float32 vectors in the same order. Kept module-level so it pickles.
    """
    (
        model_name,
        cache_dir,
        offline_dir,
        threads,
        arena,
        max_tokens,
        batch_size,
        texts,
    ) = args
    from fastembed import TextEmbedding

    te = TextEmbedding(
        model_name,
        cache_dir=cache_dir,
        threads=threads,
        enable_cpu_mem_arena=arena,
        specific_model_path=offline_dir,
    )
    if max_tokens is not None:
        tokenizer = getattr(te.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "enable_truncation"):
            tokenizer.enable_truncation(max_length=max_tokens)
    return [
        np.asarray(v, dtype=np.float32).ravel()
        for v in te.passage_embed(list(texts), batch_size=batch_size)
    ]


class FastEmbedProvider:
    """FastEmbed-based dense provider; models download on first use.

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
    ``batch_size`` defaults to 1: one passage per inference call, so
    the per-call peak is bounded by a single truncated passage.
    ``offline_model_dir`` points at the model files bundled in the
    package (air-gapped installs): when set, the model is loaded from
    that directory directly and no download or cache lookup happens.
    """

    def __init__(
        self,
        dense_model: str,
        cache_dir: Path | None = None,
        dense: DenseModelLike | None = None,
        batch_size: int = 1,
        max_tokens: int | None = None,
        threads: int | None = None,
        enable_arena: bool = False,
        index_max_tokens: int | None = None,
        indexing_workers: int = 1,
        dense_cache_dir: Path | None = None,
        offline_model_dir: Path | None = None,
    ) -> None:
        self._dense_model = dense_model
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._threads = threads
        self._enable_arena = enable_arena
        self._index_max_tokens = index_max_tokens
        self._indexing_workers = indexing_workers
        self._dense_cache_dir = dense_cache_dir
        self._offline_model_dir = offline_model_dir
        self._dense: DenseModelLike | None = dense
        self._cache_dir = cache_dir

    def _ensure_dense(self) -> None:
        """Lazily load the dense model (and only the dense model).

        When ``offline_model_dir`` is set (a model bundled in the
        package, air-gapped installs), fastembed loads the weights and
        tokenizer from that directory directly and never touches the
        network or the download cache.
        """
        if self._dense is None:
            from fastembed import TextEmbedding

            cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
            offline_dir = (
                str(self._offline_model_dir)
                if self._offline_model_dir is not None
                else None
            )
            if offline_dir is not None:
                logger.info(
                    "loading dense model %s from bundled assets %s",
                    self._dense_model,
                    offline_dir,
                )
            else:
                logger.info("loading dense model %s", self._dense_model)
            self._dense = TextEmbedding(
                self._dense_model,
                cache_dir=cache_dir,
                threads=self._threads,
                enable_cpu_mem_arena=self._enable_arena,
                specific_model_path=offline_dir,
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

    def _tokenizer(self) -> object | None:
        model = getattr(self._dense, "model", None)
        tokenizer = getattr(model, "tokenizer", None)
        return tokenizer

    def _prepare(self, text: str) -> tuple[str, int]:
        """Truncate a passage to ``index_max_tokens`` (speed/memory).

        Returns ``(truncated_text, token_len)``. Inert (returns the text
        unchanged with length 0) when no cap is set or the model exposes
        no tokenizer (test fakes), so behaviour then is unchanged.
        """
        if self._index_max_tokens is None:
            return text, 0
        tokenizer = self._tokenizer()
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return text, 0
        tok: Any = tokenizer
        enc = tok.encode(text)
        ids = list(enc.ids)
        if len(ids) > self._index_max_tokens:
            ids = ids[: self._index_max_tokens]
        truncated: str = tok.decode(ids, skip_special_tokens=True)
        return truncated, len(ids)

    def _cache_key(self, text: str) -> str:
        """Content-addressed key: the vector is a pure function of the
        model and the (already truncated) passage text — batch order and
        padding are masked out, so they do not affect it."""
        if self._dense_cache_dir is None:
            return ""
        h = hashlib.sha256()
        h.update(self._dense_model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_load(self, key: str) -> np.ndarray | None:
        if self._dense_cache_dir is None or not key:
            return None
        path = self._dense_cache_dir / key[:2] / f"{key}.npy"
        if not path.exists():
            return None
        try:
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float32)
        except Exception as exc:  # corrupt/absent -> recompute
            logger.debug("dense cache read failed for %s: %s", key[:12], exc)
            return None

    def _cache_store(self, key: str, vec: np.ndarray) -> None:
        if self._dense_cache_dir is None or not key:
            return
        path = self._dense_cache_dir / key[:2] / f"{key}.npy"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vec)
        except Exception as exc:
            logger.warning("dense vector cache write failed: %s", exc)

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Serial embedding of an already length-sorted text list."""
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
        return vectors

    def _embed_parallel(self, items: list[tuple[str, int]]) -> list[np.ndarray]:
        """Data-parallel embedding (N spawn workers, each its own model).

        ``items`` are length-sorted ``(text, token_len)`` pairs; they are
        split into contiguous slices so each worker's batches stay
        length-uniform (low padding). Results are concatenated in slice
        order, preserving the length-sorted order.
        """
        texts = [t for t, _ in items]
        n = min(self._indexing_workers, len(texts))
        if n <= 1:
            return self._embed_texts(texts)
        slices = [texts[start:end] for start, end in _even_slices(len(texts), n)]
        args = [
            (
                self._dense_model,
                str(self._cache_dir) if self._cache_dir is not None else None,
                str(self._offline_model_dir)
                if self._offline_model_dir is not None
                else None,
                self._threads,
                self._enable_arena,
                self._max_tokens,
                self._batch_size,
                s,
            )
            for s in slices
        ]
        from multiprocessing import get_context

        ctx = get_context("spawn")
        with ctx.Pool(processes=len(slices)) as pool:
            outs = pool.map(_embed_slice, args)
        flat: list[np.ndarray] = []
        for o in outs:
            flat.extend(o)
        return flat

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

        # #4: pre-truncate each passage to index_max_tokens.
        prepared = [self._prepare(t) for t in texts]

        # #2: content-addressed cache lookup (reindex / repeated content
        # and shared data_dir become cache hits).
        results: list[np.ndarray | None] = [None] * len(prepared)
        keys: list[str] = []
        misses: list[tuple[int, str, int]] = []
        for i, (text, length) in enumerate(prepared):
            key = self._cache_key(text)
            keys.append(key)
            cached = self._cache_load(key)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, text, length))

        if misses:
            # #1: length-aware batching — sort misses by token length so
            # each batch pads to a similar length (less wasted compute).
            misses.sort(key=lambda m: m[2])
            if self._indexing_workers > 1 and len(misses) > self._batch_size:
                ordered = self._embed_parallel([(t, length) for _, t, length in misses])
            else:
                ordered = self._embed_texts([t for _, t, _ in misses])
            if len(ordered) != len(misses):
                raise RuntimeError(
                    f"embedding produced {len(ordered)} vectors for "
                    f"{len(misses)} cache-miss texts"
                )
            for new_pos, (orig_i, _, _) in enumerate(misses):
                vec = ordered[new_pos]
                results[orig_i] = vec
                # #2: store the freshly computed vector.
                self._cache_store(keys[orig_i], vec)

        out: list[list[float]] = []
        for r in results:
            if r is None:  # defensive: every slot is populated above
                raise RuntimeError("internal: unpopulated embedding slot")
            out.append([float(x) for x in r])
        return out

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
