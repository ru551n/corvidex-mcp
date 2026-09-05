"""Tests for the embedding provider (offline: fake models, no downloads)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from corvidex_mcp.embeddings import FastEmbedProvider


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


def make_provider() -> tuple[FastEmbedProvider, FakeDense]:
    dense = FakeDense()
    return FastEmbedProvider("fake/model", cache_dir=None, dense=dense), dense


def test_dense_passage_batch_order() -> None:
    provider, dense = make_provider()
    out = provider.embed_passages(["ab", "cdef", "ghijkl"])
    assert len(out) == 3
    assert dense.passage_calls == [["ab", "cdef", "ghijkl"]]  # one batch call
    assert out[0] == [0.0, 2.0, 0.0, 0.0]
    assert out[1] == [1.0, 4.0, 0.0, 0.0]
    assert out[2] == [2.0, 6.0, 0.0, 0.0]
    assert provider.dimension == 4
    assert provider.model_name == "fake/model"


def test_dense_query() -> None:
    provider, dense = make_provider()
    vec = provider.embed_query("synchronous reset")
    assert vec == [1.0, 0.0, 0.0, 0.0]
    assert dense.query_calls == ["synchronous reset"]


def test_empty_inputs() -> None:
    provider, _ = make_provider()
    assert provider.embed_passages([]) == []


def test_dense_count_mismatch_raises() -> None:
    class ShortDense(FakeDense):
        def passage_embed(self, texts, batch_size=32):
            yield np.zeros(4, dtype=np.float32)

    provider = FastEmbedProvider("fake/model", cache_dir=None, dense=ShortDense())
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        provider.embed_passages(["a", "bb"])


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

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads=None,
        enable_cpu_mem_arena=None,
        specific_model_path: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.enable_cpu_mem_arena = enable_cpu_mem_arena
        self.specific_model_path = specific_model_path
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
    # Arena is off by default (lower resident RAM).
    assert stub.enable_cpu_mem_arena is False


def test_dense_arena_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    StubTextEmbedding.instances = []
    monkeypatch.setattr("fastembed.TextEmbedding", StubTextEmbedding)
    provider = FastEmbedProvider("fake/model", enable_arena=True)
    assert provider.embed_passages(["ab"])
    (stub,) = StubTextEmbedding.instances
    assert stub.enable_cpu_mem_arena is True


def test_dense_no_token_cap_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    StubTextEmbedding.instances = []
    monkeypatch.setattr("fastembed.TextEmbedding", StubTextEmbedding)
    provider = FastEmbedProvider("fake/model")
    assert provider.embed_passages(["ab"])
    (stub,) = StubTextEmbedding.instances
    assert stub.threads is None
    assert stub.model.tokenizer.caps == []


def test_dense_batch_size_default_is_bounded() -> None:
    provider, _ = make_provider()
    # Default 1: one passage per inference call — peak inference memory
    # bounded by a single truncated passage.
    assert provider._batch_size == 1


class _WordTokens:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids


class _WordTokenizer:
    """Whitespace tokenizer: one token per word (offline stand-in)."""

    def __init__(self) -> None:
        self.caps: list[int] = []

    def enable_truncation(self, max_length: int) -> None:
        self.caps.append(max_length)

    def encode(self, text: str):
        return _WordTokens(text.split())

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


class _WordModel:
    def __init__(self) -> None:
        self.tokenizer = _WordTokenizer()


class WordFakeDense(FakeDense):
    """FakeDense exposing a ``.model.tokenizer`` (word-level)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _WordModel()


def _word_provider(
    dense_cache_dir: Path | None = None,
    index_max_tokens: int = 16,
    indexing_workers: int = 1,
    batch_size: int = 8,
) -> tuple[FastEmbedProvider, WordFakeDense]:
    dense = WordFakeDense()
    provider = FastEmbedProvider(
        "fake/model",
        cache_dir=None,
        dense=dense,
        index_max_tokens=index_max_tokens,
        indexing_workers=indexing_workers,
        batch_size=batch_size,
        dense_cache_dir=dense_cache_dir,
    )
    return provider, dense


def test_index_truncation_at_token_cap() -> None:
    provider, dense = _word_provider(index_max_tokens=2)
    out = provider.embed_passages(["alpha beta gamma"])
    # passage_embed sees the truncated text (2 of 3 tokens).
    assert dense.passage_calls == [["alpha beta"]]
    assert len(out) == 1


def test_length_aware_batch_ordering() -> None:
    provider, dense = _word_provider(index_max_tokens=16)
    out = provider.embed_passages(["x y z", "a", "p q"])  # 3, 1, 2 tokens
    # Misses are sent length-sorted ...
    assert dense.passage_calls == [["a", "p q", "x y z"]]
    # ... but results map back to the original order.
    assert out[0] == [2.0, 5.0, 0.0, 0.0]  # "x y z"
    assert out[1] == [0.0, 1.0, 0.0, 0.0]  # "a"
    assert out[2] == [1.0, 3.0, 0.0, 0.0]  # "p q"


def test_dense_vector_cache_hit(tmp_path) -> None:
    provider, dense = _word_provider(dense_cache_dir=tmp_path / "dc")
    first = provider.embed_passages(["hello world", "second doc"])
    assert len(dense.passage_calls) == 1  # all misses on a cold cache
    second = provider.embed_passages(["hello world", "second doc"])
    assert second == first  # vectors identical
    assert len(dense.passage_calls) == 1  # both now cache hits
    assert len(list((tmp_path / "dc").rglob("*.npy"))) == 2


def test_dense_vector_cache_isolated_per_model(tmp_path) -> None:
    cache = tmp_path / "dc"
    p1, _ = _word_provider(dense_cache_dir=cache)
    p1._dense_model = "fake/model-a"
    p1.embed_passages(["same text"])
    p2, d2 = _word_provider(dense_cache_dir=cache)
    p2._dense_model = "fake/model-b"
    out = p2.embed_passages(["same text"])
    assert len(d2.passage_calls) == 1  # no cross-model cache reuse
    assert len(out) == 1


def test_dense_vector_cache_corrupt_recomputes(tmp_path) -> None:
    cache = tmp_path / "dc"
    provider, dense = _word_provider(dense_cache_dir=cache)
    provider.embed_passages(["hello world"])
    (vec_path,) = list(cache.rglob("*.npy"))
    vec_path.write_bytes(b"not a numpy array")
    out = provider.embed_passages(["hello world"])
    assert len(dense.passage_calls) == 2  # recomputed after corruption
    assert len(out) == 1
    # The corrupted entry was rewritten with valid data.
    assert np.load(vec_path).shape[0] == 4


def test_parallel_fallback_when_misses_small() -> None:
    # workers > 1 but misses <= batch_size stay on the serial path
    # (no pool spawn).
    provider, dense = _word_provider(indexing_workers=4, batch_size=8)
    out = provider.embed_passages(["a", "b c", "d e f"])
    assert dense.passage_calls == [["a", "b c", "d e f"]]
    assert len(out) == 3


def test_even_slices_balanced() -> None:
    from corvidex_mcp.embeddings.provider import _even_slices

    assert _even_slices(10, 4) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert _even_slices(3, 8) == [(0, 1), (1, 2), (2, 3)]  # workers > count
    assert _even_slices(0, 4) == []
    assert _even_slices(10, 1) == [(0, 10)]
    # No item is skipped or duplicated.
    for count, workers in [(1089, 8), (11, 4), (1, 1)]:
        flat = [
            i for start, end in _even_slices(count, workers) for i in range(start, end)
        ]
        assert flat == list(range(count))
