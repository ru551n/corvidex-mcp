"""Embedding layer.

Hides the concrete local embedding implementation (FastEmbed) behind a
small interface so the rest of the codebase only deals in lists of floats.
Queries and indexed passages use different entry points: jina-embeddings-v2
models require task prefixes ("query:"/"passage:"), which FastEmbed applies
through ``query_embed``/``passage_embed``.
"""

from .provider import EmbeddingProvider, FastEmbedProvider

__all__ = ["EmbeddingProvider", "FastEmbedProvider"]
