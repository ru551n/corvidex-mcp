"""Tests for the MCP server: tool registration, tool calls, sync
containment, and the single-instance lock.

Runs fully offline: local file:// git remotes and fake embedding
providers. The healthy fixture repository contains only docs/code
files, so no LSP server is spawned.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from vhdl_rag_mcp.config import AppConfig, RepositoryConfig
from vhdl_rag_mcp.embeddings.provider import FastEmbedProvider
from vhdl_rag_mcp.embeddings.providers import EmbeddingProviders
from vhdl_rag_mcp.models import (
    INDEX_SCHEMA_VERSION,
    Chunk,
    CollectionName,
    ContentType,
)
from vhdl_rag_mcp.server import (
    VhdlRagApp,
    _acquire_lock,
    config_from_args,
    create_mcp,
)
from vhdl_rag_mcp.state import StateStore

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}

STD_MD = "# Standard\n\n## Reset conventions\n\nAsync resets are named rst_n.\n"
FIFO_C = "int fifo_write(int *mem) {\n    return 0;\n}\n"


def make_hdl_chunk(file: str, language: str, kind: str, commit: str) -> Chunk:
    return Chunk(
        repository="repo",
        branch="main",
        commit=commit,
        file=file,
        content_type=ContentType.SOURCE,
        language=language,
        collection=CollectionName.HDL,
        symbol=file.rsplit("/", 1)[-1].replace(".", "_"),
        symbol_kind=kind,
        start_line=1,
        end_line=5,
        content=f"{language} fifo body referencing FIFO_DEPTH\n" * 3,
        symbols=("FIFO_DEPTH",),
    )


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# -- fake embeddings -------------------------------------------------------


class FakeDense:
    embedding_size = 4

    def passage_embed(self, texts, batch_size=32):
        for i, text in enumerate(texts):
            yield np.array([float(len(text)), float(i), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, query, batch_size=32):
        yield np.array([float(len(query)), 0.0, 0.0, 0.0], dtype=np.float32)


def fake_providers(config: AppConfig) -> EmbeddingProviders:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    return providers


@pytest.fixture
async def env(tmp_path: Path):
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "docs").mkdir()
    (up / "docs" / "standard.md").write_text(STD_MD)
    (up / "src").mkdir()
    (up / "src" / "fifo.c").write_text(FIFO_C)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")

    # "broken" (same remote, nonexistent ref) exercises per-repository
    # error containment: its failure must not affect "repo".
    config = AppConfig(
        data_dir=tmp_path / "data",
        sync_interval=10,
        repositories=[
            RepositoryConfig(name="repo", url=str(up), ref="main"),
            RepositoryConfig(name="broken", url=str(up), ref="no-such-branch"),
        ],
    )
    providers = fake_providers(config)
    app = VhdlRagApp(config, providers=providers)
    app.ensure_collections()
    await app.sync_all()
    mcp = create_mcp(app)
    yield app, mcp, up
    app.close()


def tool_text(result) -> str:
    """call_tool returns (content blocks, structured output)."""
    blocks, _structured = result
    assert blocks
    return blocks[0].text


# -- tests ---------------------------------------------------------------------


async def test_tools_registered(env) -> None:
    _app, mcp, _up = env
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {
        "search_hdl",
        "search_vhdl",
        "search_docs",
        "search_code",
        "search_knowledge",
        "get_source",
        "repository_status",
        "sync_repositories",
        "reindex_repository",
    }


async def test_search_tools_end_to_end(env) -> None:
    _app, mcp, _up = env
    result = await mcp.call_tool("search_docs", {"query": "reset conventions"})
    text = tool_text(result)
    assert "## [docs]" in text
    assert "Reset conventions" in text
    assert "repo:docs/standard.md" in text

    result = await mcp.call_tool("search_code", {"query": "fifo write function"})
    assert "## [code]" in tool_text(result)

    result = await mcp.call_tool("search_knowledge", {"query": "fifo", "limit": 10})
    knowledge = tool_text(result)
    assert "## [docs]" in knowledge
    assert "## [code]" in knowledge

    # Empty domain: friendly message, not an exception.
    result = await mcp.call_tool("search_vhdl", {"query": "entity fifo"})
    assert "No VHDL results" in tool_text(result)


async def test_search_hdl_tool_language_filter(env) -> None:
    app, mcp, _up = env
    commit = app.states.get("repo").indexed_commit or "abc123"
    chunks = [
        make_hdl_chunk("rtl/fifo.vhd", "vhdl", "entity", commit),
        make_hdl_chunk("tb/fifo_tb.v", "verilog", "design_unit", commit),
    ]
    dense = app.providers.embed_passages(
        CollectionName.HDL, [c.content for c in chunks]
    )
    app.store.upsert_chunks(chunks, dense)

    # No language: all HDL languages are searchable together.
    all_text = tool_text(await mcp.call_tool("search_hdl", {"query": "fifo"}))
    assert "repo:rtl/fifo.vhd" in all_text
    assert "repo:tb/fifo_tb.v" in all_text

    # Language filter: only Verilog chunks match.
    verilog_text = tool_text(
        await mcp.call_tool("search_hdl", {"query": "fifo", "language": "verilog"})
    )
    assert "repo:tb/fifo_tb.v" in verilog_text
    assert "repo:rtl/fifo.vhd" not in verilog_text

    # search_vhdl is the VHDL-only form of search_hdl.
    vhdl_text = tool_text(await mcp.call_tool("search_vhdl", {"query": "fifo"}))
    assert "repo:rtl/fifo.vhd" in vhdl_text
    assert "repo:tb/fifo_tb.v" not in vhdl_text

    # Cross-referencing: both chunks reference FIFO_DEPTH.
    sym_text = tool_text(
        await mcp.call_tool("search_hdl", {"query": "fifo", "symbols": ["FIFO_DEPTH"]})
    )
    assert "repo:rtl/fifo.vhd" in sym_text
    assert "repo:tb/fifo_tb.v" in sym_text
    # An identifier no chunk references yields the empty message.
    none_text = tool_text(
        await mcp.call_tool(
            "search_hdl", {"query": "fifo", "symbols": ["no_such_ident"]}
        )
    )
    assert "No HDL results" in none_text


async def test_search_hdl_language_validation(env) -> None:
    _app, mcp, _up = env
    result = await mcp.call_tool(
        "search_hdl", {"query": "fifo", "language": "verilog-2005"}
    )
    assert tool_text(result).startswith("Error: unknown HDL language")
    result = await mcp.call_tool("search_hdl", {"query": "fifo", "language": "  "})
    assert tool_text(result).startswith("Error: language must not be empty")


async def test_repository_status_analyzer_available(env, tmp_path: Path) -> None:
    from fake_veridian_util import fake_veridian as make_fake_veridian

    app, _mcp, _up = env
    # A resolvable fake Veridian: the status reports lsp mode + version.
    veridian = make_fake_veridian(tmp_path, "veridian", {})
    config = app.config.model_copy(update={"veridian_path": str(veridian)})
    app2 = VhdlRagApp(
        config, providers=app.providers, store=app.store, states=app.states
    )
    mcp2 = create_mcp(app2)
    try:
        result = await mcp2.call_tool("repository_status", {})
    finally:
        app2.close()
    text = tool_text(result)
    assert "- veridian: lsp, veridian 9.9.9-test" in text
    assert str(veridian) in text


async def test_search_tool_errors(env) -> None:
    _app, mcp, _up = env
    result = await mcp.call_tool(
        "search_docs", {"query": "x", "repository": "no-such-repo"}
    )
    assert tool_text(result).startswith("Error: unknown repository")
    result = await mcp.call_tool("search_docs", {"query": "   "})
    assert tool_text(result).startswith("Error: query must not be empty")


async def test_get_source_tool(env) -> None:
    _app, mcp, _up = env
    result = await mcp.call_tool(
        "get_source", {"repository": "repo", "file": "src/fifo.c"}
    )
    text = tool_text(result)
    assert text.startswith("repo:src/fifo.c @ ")
    assert FIFO_C.rstrip() in text

    result = await mcp.call_tool(
        "get_source",
        {"repository": "repo", "file": "src/fifo.c", "start_line": 2, "end_line": 2},
    )
    assert "(lines 2-2" in tool_text(result)

    result = await mcp.call_tool("get_source", {"repository": "repo", "file": "no.c"})
    assert tool_text(result).startswith("Error:")


async def test_repository_status_tool(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, mcp, _up = env
    # Deterministic analyzer discovery: nothing on PATH.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    result = await mcp.call_tool("repository_status", {})
    text = tool_text(result)
    assert "- repo (ref main, priority 1, domains: hdl, docs, code)" in text
    assert "indexed:" in text
    # Chunk and file counts are reported per domain with a total.
    assert "chunks: " in text
    assert " total)" in text
    assert "files:" in text
    # The healthy repo has no error; the broken one does.
    repo_block = text.split("- repo (")[1].split("- broken")[0]
    assert "last error" not in repo_block
    # Nothing was indexed for the broken repo.
    broken_block = text.split("- broken (")[1]
    assert "chunks: 0 hdl + 0 docs + 0 code (0 total)" in broken_block
    assert "last error:" in broken_block
    assert "never" in broken_block.split("\n")[1]
    # The HDL analyzer section: both analyzers in fallback mode.
    assert "HDL analyzers:" in text
    assert "- vhdl_ls: fallback" in text
    assert "- veridian: fallback" in text
    assert "was not found" in text


async def test_sync_repositories_contains_errors(env) -> None:
    _app, mcp, _up = env
    result = await mcp.call_tool(
        "sync_repositories", {"repositories": ["repo", "broken"]}
    )
    text = tool_text(result)
    assert "- repo: ok" in text
    assert "- broken: ERROR" in text
    assert "does not resolve" in text

    # Selecting only the healthy repo works and is idempotent.
    result = await mcp.call_tool("sync_repositories", {"repositories": ["repo"]})
    text = tool_text(result)
    assert "- repo: ok" in text
    assert "broken" not in text

    # Unknown names are rejected up front.
    result = await mcp.call_tool("sync_repositories", {"repositories": ["ghost"]})
    assert tool_text(result).startswith("Error: unknown repository")


async def test_reindex_repository_tool(env) -> None:
    app, mcp, _up = env
    before = app.store.count()
    result = await mcp.call_tool("reindex_repository", {"repository": "repo"})
    assert "- repo: ok" in tool_text(result)
    assert app.store.count() == before
    result = await mcp.call_tool("reindex_repository", {"repository": "nope"})
    assert tool_text(result).startswith("Error: unknown repository")


async def _until(condition, timeout: float = 10.0) -> None:
    """Poll ``condition`` until truthy or ``timeout`` elapses."""
    deadline = asyncio.get_event_loop().time() + timeout
    while not condition():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


async def test_local_poll_syncs_on_change(tmp_path: Path) -> None:
    """The fast poller syncs a local working repository when its
    working-tree fingerprint changes, and stays quiet while it is stable.
    Remote repositories (none configured here) are never touched by it.
    """
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    (work / "docs.md").write_text(STD_MD)
    git(work, "add", "-A")
    git(work, "commit", "-qm", "first")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_sync_interval=1,
        repositories=[RepositoryConfig(name="work", path=work)],
    )
    app = VhdlRagApp(config, providers=fake_providers(config))
    app.ensure_collections()
    await app.sync_all()
    base = app.store.count()
    assert base > 0

    calls: list[str] = []
    original = app.pipeline.sync_repository

    async def spy(cfg):
        # Record completion, not start: the poller runs syncs as detached
        # tasks, so asserting on store state after a *started* sync races
        # the in-flight sync and flakes on slow runners.
        await original(cfg)
        calls.append(cfg.name)

    app.pipeline.sync_repository = spy
    poll = asyncio.create_task(app.local_poll())
    try:
        # A full cycle with no changes: no sync is triggered.
        await asyncio.sleep(1.5)
        assert calls == []
        # A tracked edit changes the fingerprint: the poller syncs.
        (work / "docs.md").write_text(STD_MD + "\nMore docs\n")
        await _until(lambda: calls)
        assert calls == ["work"]
        assert app.store.count() >= base
        fp_after = app.states.get("work").local_fingerprint
        assert fp_after is not None
        # The fingerprint is now current: the next cycle is quiet again.
        calls.clear()
        await asyncio.sleep(1.5)
        assert calls == []
        # A new untracked file changes the fingerprint too.
        (work / "extra.md").write_text("# Extra\n\nUntracked docs.\n")
        await _until(lambda: calls)
        assert calls == ["work"]
        assert app.store.count() > base
        assert app.states.get("work").local_fingerprint != fp_after
    finally:
        poll.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll
    app.close()


async def test_local_poll_disabled_without_local_repos(env) -> None:
    """No local repositories configured: the poller exits immediately."""
    app, _mcp, _up = env
    assert not app._has_local_repos()
    task = asyncio.create_task(app.local_poll())
    await asyncio.wait_for(task, timeout=1.0)


async def test_lock_single_instance(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    _acquire_lock(config)
    with pytest.raises(SystemExit):
        _acquire_lock(config)


async def test_unknown_repository_name_rejected(env) -> None:
    app, _mcp, _up = env
    with pytest.raises(Exception, match="unknown repository"):
        await app.sync_all(repositories=["ghost"])


async def test_drop_unconfigured_repositories(env) -> None:
    app, _mcp, _up = env
    before = app.store.count()
    assert before > 0
    # Reconfigure without "repo" (keep "broken") -> "repo" is dropped.
    config = app.config.model_copy(
        update={"repositories": [app.config.repository("broken")]}
    )
    # Share the store (its SQLite tables belong to this data dir) and state.
    app2 = VhdlRagApp(
        config, providers=app.providers, store=app.store, states=app.states
    )
    try:
        dropped = app2.drop_unconfigured_repositories()
        assert dropped == ["repo"]
        assert app2.store.count() == 0  # only "repo" had chunks
        assert "repo" not in [st.name for st in app2.states.all()]
        # Idempotent.
        assert app2.drop_unconfigured_repositories() == []
    finally:
        app2.close()


async def test_migrate_index_is_a_noop_on_current_layout(env) -> None:
    app, _mcp, _up = env
    assert app.states.schema_version == INDEX_SCHEMA_VERSION
    assert not app.states.needs_migration
    assert app.migrate_index() is False


async def test_migrate_index_migrates_a_v1_deployment(env) -> None:
    app, _mcp, _up = env
    # Simulate a v1 deployment: a flat (pre-schema) state document.
    path = app.config.state_dir / "repositories.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "repo": {"name": "repo", "indexed_commit": "deadbeef"},
                "broken": {"name": "broken"},
            }
        ),
        encoding="utf-8",
    )
    app.states = StateStore(app.config.sqlite_index_path, path)
    assert app.states.needs_migration

    assert app.migrate_index() is True
    assert app.states.get("repo").indexed_commit is None
    assert app.states.get("broken").indexed_commit is None
    assert not app.states.needs_migration
    # The imported document is marked as consumed (the import runs once).
    assert (app.config.state_dir / "repositories.json.migrated").exists()
    assert app.migrate_index() is False  # idempotent


async def test_migrate_index_drops_legacy_vhdl_collection(env) -> None:
    app, _mcp, _up = env
    # Simulate a v1 deployment: the legacy table exists and the
    # state document predates the schema version.
    conn = app.store._conn
    conn.execute("CREATE TABLE chunks_vhdl (id TEXT PRIMARY KEY)")
    conn.execute("CREATE VIRTUAL TABLE vec_vhdl USING vec0(embedding float[4])")
    conn.execute("CREATE VIRTUAL TABLE fts_vhdl USING fts5(content)")
    path = app.config.state_dir / "repositories.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"repo": {"name": "repo", "indexed_commit": "deadbeef"}}),
        encoding="utf-8",
    )
    app.states = StateStore(app.config.sqlite_index_path, path)

    assert app.migrate_index() is True
    assert "vhdl" not in app.store._tables()
    assert "hdl" in app.store._tables()


CLI_CONFIG = """\
data_dir = "~/.local/share/vhdl-rag"
sync_interval = 120
log_level = "WARNING"

[[repositories]]
name = "cli-repo"
url = "git@github.com:co/cli.git"
ref = "main"
"""


async def test_cli_config_flag_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CLI_CONFIG, encoding="utf-8")
    cfg = config_from_args(["--config", str(path)])
    assert cfg.sync_interval == 120
    assert cfg.log_level == "WARNING"
    assert [r.name for r in cfg.repositories] == ["cli-repo"]
    # Command-line overrides win over the file.
    cfg = config_from_args(
        ["--config", str(path), "--sync-interval", "60", "--log-level", "DEBUG"]
    )
    assert cfg.sync_interval == 60
    assert cfg.log_level == "DEBUG"
    assert [r.name for r in cfg.repositories] == ["cli-repo"]
    # --vhdl-ls-path/--veridian-path override the analyzer binaries.
    cfg = config_from_args(
        [
            "--config",
            str(path),
            "--vhdl-ls-path",
            "/opt/vhdl_ls/bin/vhdl_ls",
            "--veridian-path",
            "/opt/veridian/bin/veridian",
        ]
    )
    assert cfg.vhdl_ls_path == "/opt/vhdl_ls/bin/vhdl_ls"
    assert cfg.veridian_path == "/opt/veridian/bin/veridian"


async def test_cli_config_env_var(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "alt.toml"
    path.write_text(CLI_CONFIG, encoding="utf-8")
    monkeypatch.setenv("VHDL_RAG_MCP_CONFIG", str(path))
    cfg = config_from_args([])
    assert [r.name for r in cfg.repositories] == ["cli-repo"]
    # --config beats the env var.
    other = tmp_path / "other.toml"
    other.write_text('data_dir = "d2"\n', encoding="utf-8")
    cfg = config_from_args(["--config", str(other)])
    assert cfg.repositories == []


async def test_cli_overrides_are_revalidated(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CLI_CONFIG, encoding="utf-8")
    with pytest.raises(ValidationError):
        config_from_args(["--config", str(path), "--sync-interval", "5"])
