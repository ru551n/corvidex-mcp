"""Air-gapped acceptance test: the full index -> search path, zero network.

Proves the server operates air-gapped: with every socket connection
blocked (any network attempt fails hard), the process must still

1. load the dense model with zero downloads: from the model files
   bundled in the package (``assets/``, provisioned by
   ``tools/bundle_model.py``), or, when the assets are not present,
   from a pre-provisioned local cache (``CORVIDEX_EMBED_CACHE``),
2. pass the startup self-check (runtime components + models),
3. sync a local working repository through the real pipeline (real
   embedding, real sqlite-vec + FTS5 store), and
4. answer hybrid, semantic, and lexical searches.

The bundled-model path needs nothing extra:

    uv run pytest tests/test_airgap.py -v

The cache fallback (assets not provisioned into the source tree):

    CORVIDEX_EMBED_CACHE=/path/to/embed-cache uv run pytest tests/test_airgap.py -v
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest
from capability import sqlite_extensions_supported

from corvidex_mcp.config import AppConfig, EmbeddingsConfig, RepositoryConfig
from corvidex_mcp.embeddings.assets import bundled_model_dir
from corvidex_mcp.models import CollectionName
from corvidex_mcp.server import VhdlRagApp

CACHE = os.environ.get("CORVIDEX_EMBED_CACHE")
BUNDLED = bundled_model_dir("jinaai/jina-embeddings-v2-small-en")

pytestmark = pytest.mark.skipif(
    not (CACHE or BUNDLED) or not sqlite_extensions_supported(),
    reason=(
        "air-gap acceptance: requires the default model either bundled in "
        "the package (src/corvidex_mcp/assets/, provisioned by "
        "tools/bundle_model.py) or a pre-provisioned fastembed cache "
        "(CORVIDEX_EMBED_CACHE), and a Python whose stdlib SQLite "
        "supports loadable extensions (sqlite-vec)"
    ),
)

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "airgap",
    "GIT_AUTHOR_EMAIL": "airgap@example.com",
    "GIT_COMMITTER_NAME": "airgap",
    "GIT_COMMITTER_EMAIL": "airgap@example.com",
}

FIFO_VHDL = (
    "library ieee;\n"
    "use ieee.std_logic_1164.all;\n"
    "\n"
    "entity fifo is\n"
    "  port (\n"
    "    clk : in std_logic;\n"
    "    rst_n : in std_logic;\n"
    "    dout : out std_logic\n"
    "  );\n"
    "end entity fifo;\n"
    "\n"
    "architecture rtl of fifo is\n"
    "begin\n"
    "  p_out : process (clk, rst_n) is\n"
    "  begin\n"
    "    if rst_n = '0' then\n"
    "      dout <= '0';\n"
    "    end if;\n"
    "  end process p_out;\n"
    "end architecture rtl;\n"
)
STANDARD_MD = (
    "# Standard\n\n## Resets\n\nAsynchronous resets are active-low and named rst_n.\n"
)
FIFO_C = "int fifo_write(int *mem) {\n    return mem[0];\n}\n"


@pytest.fixture
def blocked_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every socket connection hard: any network access raises."""

    def _connect(*_args: object, **_kwargs: object) -> None:
        raise OSError("network disabled by the air-gap acceptance test")

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "rtl").mkdir(parents=True)
    (repo / "rtl" / "fifo.vhd").write_text(FIFO_VHDL, encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "standard.md").write_text(STANDARD_MD, encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "fifo.c").write_text(FIFO_C, encoding="utf-8")
    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "first"],
    ):
        subprocess.run(args, cwd=repo, env=GIT_ENV, capture_output=True, check=True)
    return repo


async def test_air_gapped_index_and_search(
    tmp_path: Path, blocked_network: None
) -> None:
    repo = _make_repo(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    if BUNDLED is None:
        # Model not bundled in the package: provision it via the local
        # fastembed cache instead.
        assert CACHE is not None
        (data_dir / "embed-cache").symlink_to(Path(CACHE))
    config = AppConfig(
        data_dir=data_dir,
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        embeddings=EmbeddingsConfig(),
        repositories=[RepositoryConfig(name="repo", path=repo)],
    )
    app = VhdlRagApp(config)
    try:
        # Model loads from the local cache: no download possible (blocked).
        app.ensure_collections()
        app.migrate_index()
        check = app.selfcheck()
        assert check.required_ok, check.summary()
        by_name = {c.name: c for c in check.components}
        for collection in (
            CollectionName.HDL,
            CollectionName.DOCS,
            CollectionName.CODE,
        ):
            assert by_name[f"model:{collection.value}"].ok, check.summary()

        # Real indexing (embedding of every chunk) with zero network.
        reports = await app.sync_all()
        assert all(r.get("status") == "ok" for r in reports), reports
        assert app.store.count() > 0

        # All three search modes answer offline.
        hybrid = app.retrieval.search_knowledge("fifo reset", mode="hybrid")
        assert hybrid
        semantic = app.retrieval.search_knowledge("fifo reset", mode="semantic")
        assert semantic
        lexical = app.retrieval.search_knowledge("std_logic rst_n", mode="lexical")
        assert lexical
        assert any(r.file.endswith("fifo.vhd") for r in lexical)
    finally:
        app.close()
