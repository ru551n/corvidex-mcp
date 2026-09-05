"""Cross-encoder reranking: a second-stage precision pass.

Reranks the (over-fetched) candidates RRF fusion/BM25/cosine similarity
already ranked, via FastEmbed's ``TextCrossEncoder`` - the same local
ONNX/fastembed infrastructure the dense embedding providers use, no
separate server or dependency.

Reranking is a quality *enhancement* over an already-working retrieval
result, not a requirement: unlike a missing dense embedding model
(which makes semantic search for a collection entirely non-functional
until provisioned), a reranker that fails to load or run should not
fail the search - the caller falls back to the unreranked ranking (see
:meth:`corvidex_mcp.retrieval.RetrievalService._rerank`).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CrossEncoderLike(Protocol):
    """The FastEmbed cross-encoder subset this wrapper relies on."""

    def rerank(self, query: str, documents: list[str], **kwargs: Any) -> Any: ...


class CrossEncoderReranker:
    """FastEmbed cross-encoder reranker; the model loads on first use.

    ``score`` returns the model's raw relevance logits mapped through a
    sigmoid into ``(0, 1)``, so reranked scores compose with the
    existing bounded per-repository priority bonus the same way the
    RRF/cosine scores they replace do (see
    :data:`corvidex_mcp.retrieval._PRIORITY_BONUS_CAP`).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        threads: int | None = None,
        model: CrossEncoderLike | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: CrossEncoderLike | None = model

    def _ensure_model(self) -> CrossEncoderLike:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("loading reranker model %s", self._model_name)
            cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
            self._model = TextCrossEncoder(
                self._model_name, cache_dir=cache_dir, threads=self._threads
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text, sigmoid-normalized to ``(0, 1)``.

        Raises whatever the underlying model raises on load/inference
        failure (offline, model not provisioned, ...) - callers decide
        how to degrade; this class does not swallow errors.
        """
        if not texts:
            return []
        model = self._ensure_model()
        raw = list(model.rerank(query, texts))
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
