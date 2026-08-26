"""Tests for the embedding provider (offline: fake models, no downloads)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from vhdl_rag_mcp.embeddings import FastEmbedProvider
from vhdl_rag_mcp.models import SparseVectorData


@dataclass
class FakeSparseVec:
    indices: np.ndarray
    values: np.ndarray


class FakeDense:
    """Stands in for FastEmbed's TextEmbedding."""

    embedding_size = 4

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def passage_embed(self, texts, batch_size=32):
        self.passage_calls.append(list(texts))
        for i, text in enumerate(texts):
            yield np.array([float(i), float(len(text)), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, text, batch_size=32):
        self.query_calls.append(text)
        yield np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeSparse:
    """Stands in for FastEmbed's SparseTextEmbedding."""

    def __init__(self) -> None:
        self.passage_calls: list[tuple[list[str], str]] = []
        self.query_calls: list[tuple[str, str]] = []

    def passage_embed(self, texts, mode="passage"):
        self.passage_calls.append((list(texts), mode))
        for text in texts:
            yield FakeSparseVec(
                indices=np.array([len(text), 42], dtype=np.int32),
                values=np.array([1.0, 2.0], dtype=np.float32),
            )

    def query_embed(self, text, mode="query"):
        self.query_calls.append((text, mode))
        yield FakeSparseVec(
            indices=np.array([len(text)], dtype=np.int32),
            values=np.array([1.0], dtype=np.float32),
        )


def make_provider() -> tuple[FastEmbedProvider, FakeDense, FakeSparse]:
    dense, sparse = FakeDense(), FakeSparse()
    return (
        FastEmbedProvider("fake/model", cache_dir=None, dense=dense, sparse=sparse),
        dense,
        sparse,
    )


def test_dense_passage_batch_order() -> None:
    provider, dense, _ = make_provider()
    out = provider.embed_passages(["ab", "cdef", "ghijkl"])
    assert len(out) == 3
    assert dense.passage_calls == [["ab", "cdef", "ghijkl"]]  # one batch call
    assert out[0] == [0.0, 2.0, 0.0, 0.0]
    assert out[1] == [1.0, 4.0, 0.0, 0.0]
    assert out[2] == [2.0, 6.0, 0.0, 0.0]
    assert provider.dimension == 4
    assert provider.model_name == "fake/model"


def test_dense_query() -> None:
    provider, dense, _ = make_provider()
    vec = provider.embed_query("synchronous reset")
    assert vec == [1.0, 0.0, 0.0, 0.0]
    assert dense.query_calls == ["synchronous reset"]


def test_empty_inputs() -> None:
    provider, _, _ = make_provider()
    assert provider.embed_passages([]) == []
    assert provider.embed_sparse_passages([]) == []


def test_sparse_passages() -> None:
    provider, _, sparse = make_provider()
    out = provider.embed_sparse_passages(["ab", "cdef"])
    assert len(out) == 2
    assert isinstance(out[0], SparseVectorData)
    assert out[0].indices == (2, 42)
    assert out[0].values == (1.0, 2.0)
    assert out[1].indices == (4, 42)
    # mode is always "passage" for indexed content
    assert sparse.passage_calls == [(["ab", "cdef"], "passage")]


def test_sparse_query() -> None:
    provider, _, sparse = make_provider()
    out = provider.embed_sparse_query("fifo write")
    assert out.indices == (10,)
    assert out.values == (1.0,)
    assert sparse.query_calls == [("fifo write", "query")]


def test_sparse_query_empty_is_empty_vector() -> None:
    class NoResults:
        def query_embed(self, text, mode="query"):
            return iter(())

        def passage_embed(self, texts, mode="passage"):
            return iter(())

    provider = FastEmbedProvider(
        "fake/model", cache_dir=None, dense=FakeDense(), sparse=NoResults()
    )
    out = provider.embed_sparse_query("??")
    assert out.indices == ()
    assert out.is_empty


def test_dense_count_mismatch_raises() -> None:
    class ShortDense(FakeDense):
        def passage_embed(self, texts, batch_size=32):
            yield np.zeros(4, dtype=np.float32)

    provider = FastEmbedProvider(
        "fake/model", cache_dir=None, dense=ShortDense(), sparse=FakeSparse()
    )
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        provider.embed_passages(["a", "bb"])


def test_sparse_count_mismatch_raises() -> None:
    class ShortSparse(FakeSparse):
        def passage_embed(self, texts, mode="passage"):
            yield FakeSparseVec(indices=np.array([1]), values=np.array([1.0]))

    provider = FastEmbedProvider(
        "fake/model", cache_dir=None, dense=FakeDense(), sparse=ShortSparse()
    )
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        provider.embed_sparse_passages(["a", "bb"])


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.caps: list[int] = []

    def enable_truncation(self, max_length: int) -> None:
        self.caps.append(max_length)


class _StubModel:
    def __init__(self) -> None:
        self.tokenizer = _RecordingTokenizer()


class StubTextEmbedding:
    """Captures fastembed.TextEmbedding constructor arguments (offline)."""

    instances: ClassVar[list[StubTextEmbedding]] = []

    embedding_size = 4

    def __init__(self, model_name: str, cache_dir: str | None = None, threads=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.model = _StubModel()
        StubTextEmbedding.instances.append(self)

    def passage_embed(self, texts, batch_size=8):
        for i, text in enumerate(texts):
            yield np.array([float(i), float(len(text)), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, text, batch_size=8):
        yield np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_dense_token_cap_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    StubTextEmbedding.instances = []
    monkeypatch.setattr("fastembed.TextEmbedding", StubTextEmbedding)
    provider = FastEmbedProvider("fake/model", max_tokens=512, threads=3)
    out = provider.embed_passages(["ab", "cdef"])
    assert len(out) == 2
    (stub,) = StubTextEmbedding.instances
    assert stub.threads == 3
    assert stub.model.tokenizer.caps == [512]


def test_dense_no_token_cap_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    StubTextEmbedding.instances = []
    monkeypatch.setattr("fastembed.TextEmbedding", StubTextEmbedding)
    provider = FastEmbedProvider("fake/model")
    assert provider.embed_passages(["ab"])
    (stub,) = StubTextEmbedding.instances
    assert stub.threads is None
    assert stub.model.tokenizer.caps == []


def test_dense_batch_size_default_is_bounded() -> None:
    provider, _, _ = make_provider()
    assert provider._batch_size == 8
