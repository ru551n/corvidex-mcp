"""Tests for CrossEncoderReranker (fake model, no ONNX/network)."""

from __future__ import annotations

import math

import pytest

from corvidex_mcp.embeddings.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Returns a raw logit equal to how many times "fifo" appears."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str], **kwargs: object) -> list[float]:
        self.calls.append((query, list(documents)))
        return [float(doc.lower().count("fifo")) for doc in documents]


def test_score_empty_texts_short_circuits() -> None:
    reranker = CrossEncoderReranker("fake/model", model=FakeCrossEncoder())
    assert reranker.score("query", []) == []


def test_score_sigmoid_normalizes_to_unit_interval() -> None:
    fake = FakeCrossEncoder()
    reranker = CrossEncoderReranker("fake/model", model=fake)
    scores = reranker.score("fifo write", ["no match here", "fifo fifo fifo"])
    assert all(0.0 < s < 1.0 for s in scores)
    assert scores[1] > scores[0]
    assert fake.calls == [("fifo write", ["no match here", "fifo fifo fifo"])]


def test_score_matches_manual_sigmoid() -> None:
    fake = FakeCrossEncoder()
    reranker = CrossEncoderReranker("fake/model", model=fake)
    (score,) = reranker.score("fifo", ["fifo"])
    assert score == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_ensure_model_is_lazy_and_cached() -> None:
    fake = FakeCrossEncoder()
    reranker = CrossEncoderReranker("fake/model", model=fake)
    reranker.score("q", ["a"])
    reranker.score("q", ["b"])
    # The injected fake is reused, not reconstructed.
    assert reranker._model is fake
