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


class EmbeddingProviders:
    """Dense model per collection + one shared sparse model, lazy."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._dense: dict[str, FastEmbedProvider] = {}
        self._sparse: FastEmbedProvider | None = None

    def _dense_model_name(self, collection: CollectionName) -> str:
        emb = self._config.embeddings
        return {
            CollectionName.VHDL: emb.vhdl_model,
            CollectionName.DOCS: emb.docs_model,
            CollectionName.CODE: emb.code_model,
        }[collection]

    def _dense_provider(self, collection: CollectionName) -> FastEmbedProvider:
        name = self._dense_model_name(collection)
        provider = self._dense.get(name)
        if provider is None:
            provider = FastEmbedProvider(
                name,
                sparse_model=self._config.embeddings.sparse_model,
                cache_dir=self._config.embed_cache_dir,
            )
            self._dense[name] = provider
        return provider

    def _sparse_provider(self) -> FastEmbedProvider:
        if self._sparse is None:
            # Sparse-only provider: with independent lazy loading the dense
            # model is never touched.
            self._sparse = FastEmbedProvider(
                self._config.embeddings.sparse_model,
                self._config.embeddings.sparse_model,
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
