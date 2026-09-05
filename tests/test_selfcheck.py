"""Tests for the startup self-check and degraded startup behavior."""

from __future__ import annotations

import numpy as np
import pytest
from capability import sqlite_extensions_supported

from corvidex_mcp.config import AppConfig, RepositoryConfig
from corvidex_mcp.embeddings.provider import FastEmbedProvider
from corvidex_mcp.embeddings.providers import EmbeddingProviders
from corvidex_mcp.models import CollectionName
from corvidex_mcp.retrieval import RetrievalError
from corvidex_mcp.selfcheck import (
    ComponentStatus,
    SelfCheckResult,
    check_fts5,
    check_git,
    check_schema,
    check_sqlite,
    check_sqlite_vec,
    run_self_check,
)
from corvidex_mcp.server import VhdlRagApp
from corvidex_mcp.state import StateStore
from corvidex_mcp.vector_store import VectorStore

pytestmark = pytest.mark.skipif(
    not sqlite_extensions_supported(),
    reason=(
        "stdlib SQLite lacks loadable-extension support (the sqlite-vec "
        "extension cannot load; use CPython 3.14 or a system/homebrew "
        "Python)"
    ),
)


REQUIRED_NAMES = ("git", "sqlite", "fts5", "sqlite-vec", "schema")


class FakeDense:
    embedding_size = 4

    def passage_embed(self, texts, batch_size=32):
        for i, text in enumerate(texts):
            yield np.array([float(len(text)), float(i), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, query, batch_size=32):
        yield np.array([float(len(query)), 0.0, 0.0, 0.0], dtype=np.float32)


def make_app(tmp_path, failing: frozenset[CollectionName] = frozenset()) -> VhdlRagApp:
    """A fully wired app on a local git remote; the collections in
    ``failing`` simulate embedding models that cannot load."""
    up = tmp_path / "upstream"
    up.mkdir()
    cfg = RepositoryConfig(name="repo", url=str(up), ref="main")
    config = AppConfig(
        data_dir=tmp_path / "data",
        repositories=[cfg],
    )
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    if failing:
        original_dimension = providers.dimension
        original_embed = providers.embed_query

        def _dimension(collection: CollectionName) -> int:
            if collection in failing:
                raise RuntimeError(
                    f"model for {collection.value} not found in the offline cache"
                )
            return original_dimension(collection)

        def _embed(collection: CollectionName, text: str) -> list[float]:
            if collection in failing:
                raise RuntimeError(
                    f"model for {collection.value} not found in the offline cache"
                )
            return original_embed(collection, text)

        providers.dimension = _dimension  # type: ignore[method-assign]
        providers.embed_query = _embed  # type: ignore[method-assign]
    app = VhdlRagApp(
        config,
        providers=providers,
        store=VectorStore(config),
        states=StateStore(config.sqlite_index_path),
    )
    app.ensure_collections()
    app.migrate_index()
    return app


# -- individual checks --------------------------------------------------------


def test_required_checks_pass_in_this_runtime():
    for check in (check_git(), check_sqlite(), check_fts5(), check_sqlite_vec()):
        assert check.ok, f"{check.name}: {check.detail}"
        assert not check.optional


def test_git_missing_is_fatal(monkeypatch: pytest.MonkeyPatch):
    import corvidex_mcp.selfcheck as selfcheck

    monkeypatch.setattr(selfcheck.shutil, "which", lambda _name: None)
    check = check_git()
    assert not check.ok
    assert not check.optional
    assert "not found" in check.detail


def test_schema_check(tmp_path):
    config = AppConfig(data_dir=tmp_path / "data")
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    check = check_schema(store)
    assert check.ok
    store.close()


def test_selfcheck_result_summary_and_flags():
    ok = ComponentStatus("git", True, False, "git version 2.45")
    degraded = ComponentStatus("vhdl_ls", False, True, "not found on PATH")
    fatal = ComponentStatus("fts5", False, False, "not available")
    all_ok = SelfCheckResult((ok,))
    assert all_ok.required_ok
    assert all_ok.degraded == ()
    assert all_ok.summary() == "ok"
    degraded_only = SelfCheckResult((ok, degraded))
    assert degraded_only.required_ok
    assert degraded_only.degraded == ("vhdl_ls",)
    assert degraded_only.summary() == "degraded: vhdl_ls"
    with_fatal = SelfCheckResult((ok, fatal, degraded))
    assert not with_fatal.required_ok
    assert with_fatal.degraded == ("fts5", "vhdl_ls")
    assert with_fatal.summary() == "FATAL: fts5; degraded: vhdl_ls"


# -- app-level self-check -----------------------------------------------------


def test_selfcheck_passes_on_healthy_app(tmp_path):
    app = make_app(tmp_path)
    try:
        result = run_self_check(app)
        by_name = {c.name: c for c in result.components}
        for name in REQUIRED_NAMES:
            assert by_name[name].ok, f"{name}: {by_name[name].detail}"
        for collection in (
            CollectionName.HDL,
            CollectionName.DOCS,
            CollectionName.CODE,
        ):
            assert by_name[f"model:{collection.value}"].ok
        assert result.required_ok
        # Optional components may be degraded (no analyzers in tests) but
        # never fatal.
        assert not any(not c.ok and not c.optional for c in result.components)
    finally:
        app.close()


def test_selfcheck_reuses_cached_analyzer_probe(tmp_path, monkeypatch):
    """run_self_check() must reuse app.analyzer_statuses() (memoized on
    first access) rather than probing the analyzer binaries again —
    probing spawns a subprocess per analyzer."""
    import corvidex_mcp.server as server_mod

    app = make_app(tmp_path)
    try:
        calls = 0
        real_build = server_mod.build_analyzer_statuses

        def counting_build(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_build(*args, **kwargs)

        monkeypatch.setattr(server_mod, "build_analyzer_statuses", counting_build)
        app.analyzer_statuses()
        assert calls == 1
        run_self_check(app)
        assert calls == 1
    finally:
        app.close()


def test_model_failure_degrades_the_collection(tmp_path):
    app = make_app(tmp_path, failing=frozenset({CollectionName.HDL}))
    try:
        assert app.collection_error(CollectionName.HDL) is not None
        assert app.collection_error(CollectionName.DOCS) is None
        # The degraded collection's tables are not created; the others are.
        assert app.store.count() == 0
        assert app.store._table_exists("chunks_hdl") is False
        assert app.store._table_exists("chunks_docs") is True
        assert app.store._table_exists("chunks_code") is True
        result = app.selfcheck()
        by_name = {c.name: c for c in result.components}
        assert by_name["model:hdl"].ok is False
        assert by_name["model:hdl"].optional
        assert by_name["model:docs"].ok
        assert result.required_ok
        assert "model:hdl" in result.summary()
        # Embedding search of the degraded collection is a clear error ...
        with pytest.raises(RetrievalError, match="unavailable"):
            app.retrieval.search(CollectionName.HDL, "fifo")
        # ... lexical search of it still works (no embedding involved).
        assert app.retrieval.search(CollectionName.HDL, "fifo", mode="lexical") == []
        # Cross-domain search skips the degraded collection instead of
        # failing.
        assert app.retrieval.search_knowledge("fifo") == []
    finally:
        app.close()


def test_all_models_degraded(tmp_path):
    app = make_app(
        tmp_path,
        failing=frozenset(
            {CollectionName.HDL, CollectionName.DOCS, CollectionName.CODE}
        ),
    )
    try:
        result = app.selfcheck()
        assert result.required_ok  # models are optional
        for collection in (
            CollectionName.HDL,
            CollectionName.DOCS,
            CollectionName.CODE,
        ):
            with pytest.raises(RetrievalError, match="unavailable"):
                app.retrieval.search(collection, "fifo")
        # Everything lexical still works.
        assert app.retrieval.search(CollectionName.HDL, "fifo", mode="lexical") == []
        assert app.retrieval.search_knowledge("fifo", mode="lexical") == []
    finally:
        app.close()
