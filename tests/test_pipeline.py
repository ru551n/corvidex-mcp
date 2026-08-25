"""Tests for the indexing pipeline (incremental sync, fallbacks, containment).

Runs fully offline: a local file:// git remote, a fake LSP server binary
(speaking the same framing as vhdl_ls), and fake embedding providers.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fake_lsp_util import executable_lsp_script

from vhdl_rag_mcp.config import AppConfig, RepositoryConfig
from vhdl_rag_mcp.embeddings.provider import FastEmbedProvider
from vhdl_rag_mcp.embeddings.providers import EmbeddingProviders
from vhdl_rag_mcp.git_manager import GitError, GitManager
from vhdl_rag_mcp.indexing.pipeline import IndexPipeline
from vhdl_rag_mcp.models import CollectionName
from vhdl_rag_mcp.state import StateStore
from vhdl_rag_mcp.vector_store import VectorStore

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}

FAKE_LSP = r"""#!/usr/bin/env python3
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get(b"content-length", b"0"))
    return json.loads(sys.stdin.buffer.read(length))


def send(obj):
    body = json.dumps(obj).encode()
    sys.stdout.buffer.write(
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    sys.stdout.buffer.flush()


def symbols():
    # vhdl_ls reports entity and architecture as TOP-LEVEL SIBLINGS.
    return [
        {
            "name": "entity 'fifo'",
            "kind": 2,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 4, "character": 3},
            },
            "children": [],
        },
        {
            "name": "architecture 'rtl'",
            "kind": 2,
            "range": {
                "start": {"line": 6, "character": 0},
                "end": {"line": 8, "character": 3},
            },
            "children": [
                {
                    "name": "process 'p_write'",
                    "kind": 3,
                    "range": {
                        "start": {"line": 7, "character": 4},
                        "end": {"line": 12, "character": 12},
                    },
                    "children": [],
                }
            ],
        },
    ]


read_message()  # initialize
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"capabilities": {"documentSymbolProvider": True}},
    }
)
msg = read_message()  # initialized
assert msg is not None and msg.get("method") == "initialized", msg
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        diags = (
            [
                {
                    "code": "syntax_error",
                    "message": "boom",
                    "severity": 1,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                }
            ]
            if "badfile" in uri
            else []
        )
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": diags},
            }
        )
    elif method == "textDocument/documentSymbol":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": symbols() if "badfile" not in uri else None,
            }
        )
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)
"""

FIFO_VHDL = """\
entity fifo is
  port (
    clk : in std_logic
  );
end entity fifo;

architecture rtl of fifo is
begin
end architecture rtl;
"""

BAD_VHDL = "entity broken is end broken;\n"
SIM_VHDL = "entity sim_model is end entity sim_model;\n"
STD_MD = (
    "# Standard\n\n"
    "## Reset conventions\n\n"
    "Async resets are named rst_n.\n\n"
    "```vhdl\n"
    "p_write : process (clk, rst_n) is\n"
    "begin\n"
    "  wr_ptr <= 0;\n"
    "end process;\n"
    "```\n"
)
FIFO_C = """\
#include <stdint.h>

int fifo_write(int *mem, int ptr) {
    mem[ptr] = 1;
    return 0;
}

int fifo_read(int *mem, int ptr) {
    return mem[ptr];
}
"""
NEW_VHDL = "entity new_top is end entity new_top;\n"


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


class FakeSparseVec:
    def __init__(self, indices, values) -> None:
        self.indices = np.asarray(indices, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)


class FakeSparse:
    def passage_embed(self, texts, mode="passage"):
        for text in texts:
            yield FakeSparseVec([len(text), 42], [1.0, 2.0])

    def query_embed(self, query, mode="query"):
        yield FakeSparseVec([len(query)], [1.0])


def fake_providers(config: AppConfig) -> EmbeddingProviders:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider(
        "fake/dense", "fake/sparse", dense=FakeDense(), sparse=FakeSparse()
    )
    sparse = FastEmbedProvider("fake/sparse", "fake/sparse", sparse=FakeSparse())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    providers._sparse_provider = lambda: sparse  # type: ignore[method-assign]
    return providers


# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.vhd").write_text(FIFO_VHDL)
    (up / "rtl" / "badfile.vhd").write_text(BAD_VHDL)
    (up / "sim").mkdir()
    (up / "sim" / "sim_model.vhd").write_text(SIM_VHDL)
    (up / "docs").mkdir()
    (up / "docs" / "standard.md").write_text(STD_MD)
    (up / "src").mkdir()
    (up / "src" / "fifo.c").write_text(FIFO_C)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")
    return up


@pytest.fixture
def fake_lsp(tmp_path: Path) -> Path:
    return executable_lsp_script(tmp_path, "fake_lsp", FAKE_LSP)


@pytest.fixture
def config(tmp_path: Path, remote: Path, fake_lsp: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        vhdl_ls_path=str(fake_lsp),
        repositories=[
            RepositoryConfig(
                name="repo",
                url=str(remote),
                ref="main",
            )
        ],
    )


@pytest.fixture
def env(tmp_path: Path, config: AppConfig):
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    pipeline = IndexPipeline(
        config,
        GitManager(config.repos_dir),
        store,
        fake_providers(config),
        StateStore(config.state_dir / "repositories.json"),
    )
    yield store, pipeline
    store.close()


# -- tests ---------------------------------------------------------------------


async def test_full_sync_indexes_all_domains(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    await pipeline.sync_repository(cfg)
    assert store.count() == 11  # 3+1+3 VHDL, 2 docs, 2 code
    assert store.count(CollectionName.HDL) == 7
    assert store.count(CollectionName.DOCS) == 2
    assert store.count(CollectionName.CODE) == 2
    # LSP path: the fake server's entity/architecture specs.
    vhdl = store.chunks_for_file("repo", "rtl/fifo.vhd")
    kinds = {(c.symbol_kind, c.symbol) for c in vhdl}
    assert ("entity", "fifo") in kinds
    assert ("architecture", "rtl") in kinds
    # Syntax-error file got the structural fallback.
    bad = store.chunks_for_file("repo", "rtl/badfile.vhd")
    assert [(c.symbol_kind, c.symbol) for c in bad] == [("entity", "broken")]
    # Docs: the reset section carries the fence identifiers.
    docs = store.chunks_for_file("repo", "docs/standard.md")
    reset = next(c for c in docs if c.heading == "Reset conventions")
    assert "rst_n" in reset.symbols
    assert "p_write" in reset.symbols
    # Code: one chunk per function.
    code = store.chunks_for_file("repo", "src/fifo.c")
    assert {c.symbol for c in code} == {"fifo_write", "fifo_read"}


async def test_empty_plan_when_ref_unchanged(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    await pipeline.sync_repository(cfg)
    before = store.count()
    await pipeline.sync_repository(cfg)  # ref did not move
    assert store.count() == before


async def test_incremental_add_delete_modify(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    remote = Path(cfg.url)
    await pipeline.sync_repository(cfg)
    assert store.count() == 11

    # Modify (shifts line ranges -> new chunk IDs), add, delete.
    (remote / "src" / "fifo.c").write_text(
        "#include <stdint.h>\n\nint fifo_write(int *mem, int ptr) {\n"
        "    mem[ptr] = 1;\n    return 0;\n}\n"
        "\nint fifo_read(int *mem, int ptr) {\n    return mem[ptr];\n}\n"
        "\nint fifo_reset(int *mem, int n) {\n    for (int i = 0; i < n; i++)\n"
        "        mem[i] = 0;\n}\n"
    )
    (remote / "rtl" / "new_top.vhd").write_text(NEW_VHDL)
    git(remote, "rm", "-q", "docs/standard.md")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "second")

    await pipeline.sync_repository(cfg)
    # 11 - 2 (docs deleted) - 2 (old fifo.c) + 3 (new fifo.c) + 3 (new_top
    # via LSP) = 13. Crucially the modified file's stale chunks are gone.
    assert store.count() == 13
    code = store.chunks_for_file("repo", "src/fifo.c")
    assert {c.symbol for c in code} == {"fifo_write", "fifo_read", "fifo_reset"}
    assert store.chunks_for_file("repo", "docs/standard.md") == []
    assert store.chunks_for_file("repo", "rtl/new_top.vhd") != []


async def test_incremental_rename(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    remote = Path(cfg.url)
    await pipeline.sync_repository(cfg)
    assert store.count() == 11
    git(remote, "mv", "src/fifo.c", "src/fifo2.c")
    git(remote, "commit", "-qm", "rename")
    await pipeline.sync_repository(cfg)
    assert store.count() == 11
    assert store.chunks_for_file("repo", "src/fifo.c") == []
    code = store.chunks_for_file("repo", "src/fifo2.c")
    assert {c.symbol for c in code} == {"fifo_write", "fifo_read"}


async def test_sync_error_keeps_previous_state(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    remote = Path(cfg.url)
    await pipeline.sync_repository(cfg)
    commit1 = git(remote, "rev-parse", "HEAD")

    # A second configured repo whose ref does not exist.
    broken = config.model_copy(
        update={
            "repositories": [
                *config.repositories,
                RepositoryConfig(name="bad", url=str(remote), ref="no-such-branch"),
            ]
        }
    )
    states = StateStore(broken.state_dir / "repositories.json")
    pipeline2 = IndexPipeline(
        broken, GitManager(broken.repos_dir), store, fake_providers(broken), states
    )
    with pytest.raises(GitError, match="does not resolve"):
        await pipeline2.sync_repository(broken.repository("bad"))
    assert states.get("bad").indexed_commit is None
    assert states.get("bad").last_sync_error is not None
    # The healthy repo's state is untouched.
    assert states.get("repo").indexed_commit == commit1
    assert store.count() == 11


async def test_reindex_repository(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    await pipeline.sync_repository(cfg)
    before = store.count()
    await pipeline.reindex_repository(cfg)
    assert store.count() == before
    assert store.count(CollectionName.HDL) == 7


async def test_domains_and_excludes(
    tmp_path: Path, remote: Path, fake_lsp: Path
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        vhdl_ls_path=str(fake_lsp),
        repositories=[
            RepositoryConfig(
                name="repo",
                url=str(remote),
                ref="main",
                domains=["vhdl", "code"],
                exclude=["sim"],
            )
        ],
    )
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    pipeline = IndexPipeline(
        config,
        GitManager(config.repos_dir),
        store,
        fake_providers(config),
        StateStore(config.state_dir / "repositories.json"),
    )
    await pipeline.sync_repository(config.repository("repo"))
    # docs disabled, sim excluded: fifo.vhd(3) + badfile.vhd(1) + fifo.c(2)
    assert store.count() == 6
    assert store.count(CollectionName.DOCS) == 0
    assert store.chunks_for_file("repo", "sim/sim_model.vhd") == []
    store.close()


async def test_deleted_repository_cleanup(config: AppConfig, env) -> None:
    store, pipeline = env
    cfg = config.repository("repo")
    await pipeline.sync_repository(cfg)
    assert store.count() == 11
    pipeline.delete_repository("repo")
    assert store.count() == 0


async def test_local_amend_without_content_change_advances_commit(
    tmp_path: Path, fake_lsp: Path
) -> None:
    """A content-identical HEAD move must advance the indexed commit.

    Otherwise the rewritten-away commit cannot be diffed on the next
    sync and the pipeline falls back to a full reindex of a working
    repository that did not change.
    """
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    (work / "fifo.vhd").write_text(FIFO_VHDL)
    (work / "notes.md").write_text("# N\n")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "first")
    head1 = git(work, "rev-parse", "HEAD")
    config = AppConfig(
        data_dir=tmp_path / "data",
        vhdl_ls_path=str(fake_lsp),
        repositories=[RepositoryConfig(name="work", path=work)],
    )
    store = VectorStore(config)
    store.ensure_collections(hdl_dim=4, docs_dim=4, code_dim=4)
    states = StateStore(config.state_dir / "repositories.json")
    pipeline = IndexPipeline(
        config,
        GitManager(config.repos_dir),
        store,
        fake_providers(config),
        states,
    )
    await pipeline.sync_repository(config.repository("work"))
    assert states.get("work").indexed_commit == head1
    before = store.count()

    # Amend without a content change: HEAD moves, the tree does not.
    git(work, "commit", "-q", "--amend", "-m", "rewritten")
    head2 = git(work, "rev-parse", "HEAD")
    assert head2 != head1
    await pipeline.sync_repository(config.repository("work"))
    assert store.count() == before
    # The indexed commit advanced with HEAD ...
    assert states.get("work").indexed_commit == head2
    # ... and an unchanged tree at the same commit advances nothing.
    await pipeline.sync_repository(config.repository("work"))
    assert states.get("work").indexed_commit == head2
    store.close()
