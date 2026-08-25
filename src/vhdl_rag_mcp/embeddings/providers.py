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

#: Fixed local embedding models (no configuration surface): per-collection
#: dense (jina v2, 768 dimensions) + one shared sparse BM25 model.
HDL_MODEL = "jinaai/jina-embeddings-v2-base-code"
DOCS_MODEL = "jinaai/jina-embeddings-v2-base-en"
CODE_MODEL = "jinaai/jina-embeddings-v2-base-code"
SPARSE_MODEL = "Qdrant/bm25"


class EmbeddingProviders:
    """Dense model per collection + one shared sparse model, lazy."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._dense: dict[str, FastEmbedProvider] = {}
        self._sparse: FastEmbedProvider | None = None

    def _dense_model_name(self, collection: CollectionName) -> str:
        return {
            CollectionName.HDL: HDL_MODEL,
            CollectionName.DOCS: DOCS_MODEL,
            CollectionName.CODE: CODE_MODEL,
        }[collection]

    def _dense_provider(self, collection: CollectionName) -> FastEmbedProvider:
        name = self._dense_model_name(collection)
        provider = self._dense.get(name)
        if provider is None:
            provider = FastEmbedProvider(
                name,
                sparse_model=SPARSE_MODEL,
                cache_dir=self._config.embed_cache_dir,
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
