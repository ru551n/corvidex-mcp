"""Shared test fixtures.

Autouse: forces Hugging Face Hub downloads (fastembed model fetches)
into offline mode for the whole suite by default. Every dense
embedding provider in the fast tests is already faked (``FakeDense``),
so this only matters for the newer cross-encoder reranker: with no
override, ``EmbeddingsConfig.rerank_enabled`` defaults to true, and
``RetrievalService._rerank`` degrades gracefully (falls back to the
unreranked ranking) whenever the reranker model can't load — offline
mode makes that failure immediate instead of attempting a real
download, keeping the fast suite fast, deterministic, and network-free
without needing a fake reranker injected at every construction site.

``tests/test_quality.py`` is the one place that wants the real models
and real network access (see its docstring); it opts in by setting
``CORVIDEX_RUN_QUALITY=1``, which this fixture treats as "manage your
own network access" and leaves alone.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("CORVIDEX_RUN_QUALITY") == "1":
        return
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
