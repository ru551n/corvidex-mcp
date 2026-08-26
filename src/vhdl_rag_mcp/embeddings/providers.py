"""Multi-collection embedding providers.

The three collections may use different dense models (per
:mod:`vhdl_rag_mcp.config` ``EmbeddingsConfig``) while sharing one sparse
(BM25) model. This wrapper owns the providers lazily: models are loaded
on first use and cached in the configured ``embed-cache`` directory.
"""

from __future__ import annotations

from ..config import AppConfig
from ..models import CollectionName, SparseVectorData
from .provider import FastEmbedProvider

#: Shared sparse BM25 model (dense models are per-collection and
#: configurable via the ``[embeddings]`` config section; the defaults
#: there are the jina v2 small-en model, 512 dims).
SPARSE_MODEL = "Qdrant/bm25"


class EmbeddingProviders:
    """Dense model per collection + one shared sparse model, lazy."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._dense: dict[str, FastEmbedProvider] = {}
        self._sparse: FastEmbedProvider | None = None

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
                sparse_model=SPARSE_MODEL,
                cache_dir=self._config.embed_cache_dir,
                max_tokens=eb.dense_max_tokens,
                threads=eb.dense_threads,
                enable_arena=eb.dense_enable_cpu_mem_arena,
                index_max_tokens=eb.index_max_tokens,
                indexing_workers=eb.indexing_workers,
                dense_cache_dir=self._config.dense_cache_dir,
            )
            self._dense[name] = provider
        return provider

    def _sparse_provider(self) -> FastEmbedProvider:
        if self._sparse is None:
            # Sparse-only provider: with independent lazy loading the dense
            # model is never touched.
            self._sparse = FastEmbedProvider(
                SPARSE_MODEL,
                SPARSE_MODEL,
                cache_dir=self._config.embed_cache_dir,
            )
        return self._sparse

    # -- dense (per collection) ------------------------------------------

    def dimension(self, collection: CollectionName) -> int:
        """Dense dimension for a collection (loads the model)."""
        return self._dense_provider(collection).dimension

    def embed_passages(
        self, collection: CollectionName, texts: list[str]
    ) -> list[list[float]]:
        return self._dense_provider(collection).embed_passages(texts)

    def embed_query(self, collection: CollectionName, text: str) -> list[float]:
        return self._dense_provider(collection).embed_query(text)

    # -- sparse (shared) ----------------------------------------------------

    def embed_sparse_passages(self, texts: list[str]) -> list[SparseVectorData]:
        return self._sparse_provider().embed_sparse_passages(texts)

    def embed_sparse_query(self, text: str) -> SparseVectorData:
        return self._sparse_provider().embed_sparse_query(text)
