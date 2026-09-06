"""Multi-collection embedding providers.

The three collections may use different dense models (per
:mod:`corvidex_mcp.config` ``EmbeddingsConfig``). This wrapper owns the
providers lazily: models are loaded on first use and cached in the
configured ``embed-cache`` directory.
"""

from __future__ import annotations

import asyncio

from ..config import AppConfig
from ..models import CollectionName
from .assets import bundled_model_dir
from .provider import FastEmbedProvider
from .reranker import CrossEncoderReranker


class EmbeddingProviders:
    """Dense model per collection, lazy."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._dense: dict[str, FastEmbedProvider] = {}
        self._reranker: CrossEncoderReranker | None = None

    def _dense_model_name(self, collection: CollectionName) -> str:
        e = self._config.embeddings
        return {
            CollectionName.HDL: e.hdl_model,
            CollectionName.DOCS: e.docs_model,
            CollectionName.CODE: e.code_model,
        }[collection]

    def _dense_provider(self, collection: CollectionName) -> FastEmbedProvider:
        name = self._dense_model_name(collection)
        provider = self._dense.get(name)
        if provider is None:
            eb = self._config.embeddings
            provider = FastEmbedProvider(
                name,
                cache_dir=self._config.embed_cache_dir,
                max_tokens=eb.dense_max_tokens,
                threads=eb.dense_threads,
                enable_arena=eb.dense_enable_cpu_mem_arena,
                batch_size=eb.dense_batch_size,
                index_max_tokens=eb.index_max_tokens,
                indexing_workers=eb.indexing_workers,
                dense_cache_dir=self._config.dense_cache_dir,
                offline_model_dir=bundled_model_dir(name),
            )
            self._dense[name] = provider
        return provider

    def dimension(self, collection: CollectionName) -> int:
        """Dense dimension for a collection (loads the model)."""
        return self._dense_provider(collection).dimension

    def model_name(self, collection: CollectionName) -> str:
        """Configured dense model name for a collection."""
        return self._dense_model_name(collection)

    def embed_passages(
        self, collection: CollectionName, texts: list[str]
    ) -> list[list[float]]:
        return self._dense_provider(collection).embed_passages(texts)

    def embed_query(self, collection: CollectionName, text: str) -> list[float]:
        return self._dense_provider(collection).embed_query(text)

    async def embed_passages_async(
        self, collection: CollectionName, texts: list[str]
    ) -> list[list[float]]:
        """:meth:`embed_passages`, off the event loop (ONNX inference is
        CPU-bound)."""
        return await asyncio.to_thread(self.embed_passages, collection, texts)

    async def embed_query_async(
        self, collection: CollectionName, text: str
    ) -> list[float]:
        """:meth:`embed_query`, off the event loop (ONNX inference is
        CPU-bound)."""
        return await asyncio.to_thread(self.embed_query, collection, text)

    def _rerank_provider(self) -> CrossEncoderReranker:
        if self._reranker is None:
            eb = self._config.embeddings
            self._reranker = CrossEncoderReranker(
                eb.rerank_model,
                cache_dir=self._config.embed_cache_dir,
                threads=eb.dense_threads,
            )
        return self._reranker

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder relevance score per text (see :mod:`.reranker`).

        Propagates any load/inference failure; the caller (see
        :meth:`corvidex_mcp.retrieval.RetrievalService._rerank`) decides
        how to degrade.
        """
        return self._rerank_provider().score(query, texts)

    async def rerank_async(self, query: str, texts: list[str]) -> list[float]:
        """:meth:`rerank`, off the event loop (cross-encoder inference is
        CPU-bound)."""
        return await asyncio.to_thread(self.rerank, query, texts)
