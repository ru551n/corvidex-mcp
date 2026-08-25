"""End-to-end acceptance tests.

Full app lifecycle against a local file:// git remote with fake
embedding providers: initial sync, server restart (Qdrant local
persistence + state round-trip), incremental sync from the persisted
state, and the periodic auto-sync loop. Tests gated on the
``VHDL_LS_TEST_BIN`` / ``VERIDIAN_TEST_BIN`` environment variables run
the same flows against the real language-server binaries (real LSP
handshake, diagnostics, documentSymbol).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fake_lsp_util import executable_lsp_script
from test_pipeline import FAKE_LSP

from vhdl_rag_mcp.config import AppConfig, RepositoryConfig
from vhdl_rag_mcp.embeddings.provider import FastEmbedProvider
from vhdl_rag_mcp.embeddings.providers import EmbeddingProviders
from vhdl_rag_mcp.models import CollectionName
from vhdl_rag_mcp.server import VhdlRagApp

REAL_BIN = os.environ.get("VHDL_LS_TEST_BIN")
REAL_VERIDIAN = os.environ.get("VERIDIAN_TEST_BIN")

# The remote's initial content: 1 VHDL file (entity + architecture +
# one >=5-line process), 2 docs sections, 2 C functions = 7 chunks.
FIFO_VHDL = """\
library ieee;
use ieee.std_logic_1164.all;

entity fifo is
  port (
    clk  : in  std_logic;
    data : in  std_logic_vector(7 downto 0);
    full : out std_logic
  );
end entity fifo;

architecture rtl of fifo is
  signal wr_ptr : natural range 0 to 7;
begin
  p_write : process (clk)
  begin
    if rising_edge(clk) then
      wr_ptr <= wr_ptr + 1;
      if wr_ptr = 7 then
        full <= '1';
      end if;
    end if;
  end process p_write;
end architecture rtl;
"""
STD_MD = (
    "# Standard\n\n"
    "## Reset conventions\n\n"
    "Async resets are named rst_n.\n\n"
    "```vhdl\n"
    "  if rst_n /= '1' then\n"
    "    wr_ptr <= 0;\n"
    "  end if;\n"
    "```\n"
)
FIFO_C = """\
int fifo_write(int *mem, int ptr) {
    mem[ptr] = 1;
    return 0;
}

int fifo_read(int *mem, int ptr) {
    return mem[ptr];
}
"""
NEW_VHDL = "entity new_top is end entity new_top;\n"
REAL_VERIDIAN_PKG_SV = """\
package fifo_pkg;
  localparam int FIFO_DEPTH = 8;
endpackage
"""
REAL_VERIDIAN_V = """\
module fifo (
  input logic clk,
  output logic [7:0] dout
);
  localparam int FIFO_DEPTH = 8;
  always_ff @(posedge clk) begin
    dout <= FIFO_DEPTH[7:0];
  end
endmodule
"""
MODIFIED_C = """\
int fifo_write(int *mem, int ptr) {
    mem[ptr] = 1;
    return 0;
}

int fifo_read(int *mem, int ptr) {
    return mem[ptr];
}

int fifo_reset(int *mem, int n) {
    for (int i = 0; i < n; i++)
        mem[i] = 0;
}
"""

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


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


# -- fake embeddings (module-level; cheap) ----------------------------------


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
            yield FakeSparseVec([len(text), len(text) + 7], [1.0, 2.0])

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


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.vhd").write_text(FIFO_VHDL)
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


def make_config(tmp_path: Path, remote: Path, lsp_binary: str) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        sync_interval=10,
        vhdl_ls_path=lsp_binary,
        repositories=[RepositoryConfig(name="repo", url=str(remote), ref="main")],
    )


def make_app(config: AppConfig) -> VhdlRagApp:
    app = VhdlRagApp(config, providers=fake_providers(config))
    app.ensure_collections()
    return app


# -- tests ----------------------------------------------------------------------


async def test_full_lifecycle_with_restart(
    tmp_path: Path, remote: Path, fake_lsp: Path
) -> None:
    config = make_config(tmp_path, remote, str(fake_lsp))

    # Process 1: initial full sync.
    app = make_app(config)
    await app.sync_all()
    assert app.store.count() == 7  # 3 VHDL + 2 docs + 2 code
    assert app.store.count(CollectionName.HDL) == 3
    commit1 = app.states.get("repo").indexed_commit
    assert commit1 == git(remote, "rev-parse", "HEAD")
    app.close()

    # Process 2: fresh app over the same data dir. Qdrant local mode
    # persists; state comes from the state file.
    app = make_app(config)
    assert app.store.count() == 7
    assert app.states.get("repo").indexed_commit == commit1
    results = app.retrieval.search(CollectionName.HDL, "fifo write pointer")
    assert results and results[0].repository == "repo"

    # Advance the remote: modify (new line ranges -> new chunk IDs), add,
    # delete. The incremental plan must handle all three.
    (remote / "src" / "fifo.c").write_text(MODIFIED_C)
    (remote / "rtl" / "new_top.vhd").write_text(NEW_VHDL)
    git(remote, "rm", "-q", "docs/standard.md")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "second")
    await app.sync_all()
    # 7 - 2 (docs gone) - 2 (stale fifo.c) + 3 (new fifo.c) + 3 (new_top
    # via LSP) = 9; stale chunks of the modified file are gone.
    assert app.store.count() == 9
    code = app.store.chunks_for_file("repo", "src/fifo.c")
    assert {c.symbol for c in code} == {"fifo_write", "fifo_read", "fifo_reset"}
    assert app.store.chunks_for_file("repo", "docs/standard.md") == []
    app.close()


async def test_periodic_auto_sync(
    tmp_path: Path, remote: Path, fake_lsp: Path, monkeypatch
) -> None:
    config = make_config(tmp_path, remote, str(fake_lsp))
    app = make_app(config)
    await app.sync_all()
    before = app.store.count()

    # The remote moves while the "server" is running.
    (remote / "src" / "extra.c").write_text("int extra(void) {\n    return 2;\n}\n")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "second")

    real_sleep = asyncio.sleep
    state = {"calls": 0}

    async def fake_sleep(seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] >= 2:
            # Second loop iteration: stop the periodic task.
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    task = asyncio.create_task(app.periodic_sync())
    try:
        deadline = time.monotonic() + 30
        while app.store.count() == before and time.monotonic() < deadline:
            await real_sleep(0.1)
        assert app.store.count() > before
        assert "extra" in {
            c.symbol for c in app.store.chunks_for_file("repo", "src/extra.c")
        }
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    app.close()


@pytest.mark.skipif(not REAL_BIN, reason="VHDL_LS_TEST_BIN not set")
async def test_real_lsp_end_to_end(tmp_path: Path, remote: Path) -> None:
    """Same lifecycle against the real vhdl_ls binary."""
    config = make_config(tmp_path, remote, REAL_BIN)
    app = make_app(config)
    await app.sync_all()
    try:
        # The real LSP must have produced LSP-primary chunks: entity,
        # architecture, and the process (span >= 5 lines).
        vhdl = app.store.chunks_for_file("repo", "rtl/fifo.vhd")
        kinds = {(c.symbol_kind, c.symbol) for c in vhdl}
        assert ("entity", "fifo") in kinds
        assert ("architecture", "rtl") in kinds
        assert ("process", "p_write") in kinds
        process = next(c for c in vhdl if c.symbol == "p_write")
        assert "rising_edge(clk)" in process.content
        assert process.entity == "fifo"
        assert process.architecture == "rtl"
        # Cross-domain: docs and code were chunked in the same sync.
        assert app.store.count(CollectionName.DOCS) == 2
        assert app.store.count(CollectionName.CODE) == 2
        docs = app.store.chunks_for_file("repo", "docs/standard.md")
        reset = next(c for c in docs if c.heading == "Reset conventions")
        assert "rst_n" in reset.symbols
    finally:
        app.close()


@pytest.mark.skipif(not REAL_VERIDIAN, reason="VERIDIAN_TEST_BIN not set")
async def test_real_veridian_end_to_end(tmp_path: Path) -> None:
    """Verilog/SV indexing against the real Veridian binary.

    The two fixture files share FIFO_DEPTH, so the cross-language
    cross-reference is asserted over the real index.
    """
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.v").write_text(REAL_VERIDIAN_V)
    (up / "rtl" / "fifo_pkg.sv").write_text(REAL_VERIDIAN_PKG_SV)
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")

    config = AppConfig(
        data_dir=tmp_path / "data",
        veridian_path=REAL_VERIDIAN,
        repositories=[RepositoryConfig(name="repo", url=str(up), ref="main")],
    )
    app = make_app(config)
    await app.sync_all()
    try:
        verilog = app.store.chunks_for_file("repo", "rtl/fifo.v")
        sv = app.store.chunks_for_file("repo", "rtl/fifo_pkg.sv")
        assert verilog and sv
        assert {c.language for c in verilog} == {"verilog"}
        assert {c.language for c in sv} == {"systemverilog"}
        # Real Veridian (slang) produced LSP-primary chunks: the module
        # (design_unit) and the package.
        assert any(c.symbol_kind == "design_unit" for c in verilog)
        assert any(c.symbol_kind == "package" for c in sv)
        # Cross-language cross-reference over the shared identifier.
        results = app.retrieval.search(
            CollectionName.HDL, "fifo", symbols=("FIFO_DEPTH",), limit=20
        )
        assert {r.language for r in results} == {"verilog", "systemverilog"}
    finally:
        app.close()
