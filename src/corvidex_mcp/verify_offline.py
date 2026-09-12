"""Offline self-verification: ``python -m corvidex_mcp.verify_offline``.

Shipped with the package so the *offline bundle* can prove its own
central claim on the target host: that nothing is downloaded at runtime.
The claim is worthless untested, and an air-gapped operator cannot run
this repository's pytest suite (no checkout, no dev dependencies).

What it does, in one process:

1. refuses every outbound socket connection, so a download cannot
   silently succeed — it raises instead;
2. reports which models resolve to the bundled package assets
   (``corvidex_mcp/assets/``) rather than to a cache or the network;
3. creates a throwaway repository with one VHDL file, one Markdown file
   and one C file, and indexes it through the real pipeline (real
   chunking, real ONNX embedding, real sqlite-vec + FTS5 store);
4. runs a hybrid search across all three collections with cross-encoder
   reranking enabled, and asserts the reranker actually ran.

Exit code 0 and a final ``offline verification: OK`` line mean the
installation is genuinely self-contained. Any download attempt surfaces
as a failure here rather than as a mysterious hang on a production host.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import AppConfig, EmbeddingsConfig, RepositoryConfig
from .embeddings.assets import bundled_model_dir, required_models
from .models import CollectionName

FIFO_VHDL = """\
library ieee;
use ieee.std_logic_1164.all;

entity fifo is
  port (
    clk   : in  std_logic;
    rst_n : in  std_logic;
    dout  : out std_logic
  );
end entity fifo;

architecture rtl of fifo is
begin
  p_out : process (clk, rst_n) is
  begin
    if rst_n = '0' then
      dout <= '0';
    end if;
  end process p_out;
end architecture rtl;
"""

STANDARD_MD = """\
# Standard

## Resets

Asynchronous resets are active-low and named rst_n.
"""

FIFO_C = """\
int fifo_write(int *mem) {
    return mem[0];
}
"""


def _block_network() -> None:
    """Make every outbound connection raise, for the rest of the process."""

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("network access refused by corvidex_mcp.verify_offline")

    socket.socket.connect = _refuse  # type: ignore[method-assign]
    socket.create_connection = _refuse  # type: ignore[assignment]
    for name in (
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        os.environ[name] = "1"


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "rtl").mkdir(parents=True)
    (repo / "rtl" / "fifo.vhd").write_text(FIFO_VHDL, encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "standard.md").write_text(STANDARD_MD, encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "fifo.c").write_text(FIFO_C, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "verify",
        "GIT_AUTHOR_EMAIL": "verify@example.invalid",
        "GIT_COMMITTER_NAME": "verify",
        "GIT_COMMITTER_EMAIL": "verify@example.invalid",
    }
    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "verify"],
    ):
        subprocess.run(args, cwd=repo, env=env, capture_output=True, check=True)
    return repo


def _report_models() -> list[str]:
    """One line per configured model; returns the names not bundled."""
    not_bundled: list[str] = []
    for model in required_models():
        where = bundled_model_dir(model.name)
        if where is None:
            not_bundled.append(model.name)
            print(f"  {model.name:<44} NOT bundled (would need a cache or a download)")
        else:
            print(f"  {model.name:<44} bundled: {where}")
    return not_bundled


async def _run(tmp: Path) -> None:
    from .server import VhdlRagApp

    repo = _make_repo(tmp)
    data_dir = tmp / "data"
    data_dir.mkdir()
    config = AppConfig(
        data_dir=data_dir,
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        embeddings=EmbeddingsConfig(),
        repositories=[RepositoryConfig(name="verify", path=repo)],
    )
    app = VhdlRagApp(config)
    try:
        app.ensure_collections()
        app.migrate_index()
        check = app.selfcheck()
        if not check.required_ok:
            raise SystemExit(f"self-check failed: {check.summary()}")
        by_name = {c.name: c for c in check.components}
        for collection in CollectionName:
            status = by_name[f"model:{collection.value}"]
            print(f"  model:{collection.value:<8} ok={status.ok} {status.detail or ''}")
            if not status.ok:
                raise SystemExit(f"model for {collection.value} did not load offline")

        print("indexing the throwaway repository (real embedding, no network) ...")
        reports = await app.sync_all()
        if any(r.get("status") != "ok" for r in reports):
            raise SystemExit(f"sync failed: {reports}")
        print(f"  indexed chunks: {app.store.count()}")

        print("searching each collection (hybrid + rerank) ...")
        for collection in CollectionName:
            hits = await app.retrieval.search(collection, "fifo reset", limit=3)
            if not hits:
                raise SystemExit(f"no results from the {collection.value} collection")
            top = hits[0]
            print(
                f"  {collection.value:<5} {len(hits)} hit(s), top: "
                f"{top.file} score={top.score:.3f}"
            )

        print("searching all three collections at once (fused + rerank) ...")
        results = await app.retrieval.search_knowledge("fifo reset convention", limit=9)
        if not results:
            raise SystemExit("search_knowledge returned no results")
        for result in results:
            print(f"  [{result.result_type}] {result.file} score={result.score:.3f}")

        # The reranker is the one component that degrades *silently* when
        # it cannot load (the search still answers, from the unreranked
        # ranking), so prove it ran instead of inferring it from a
        # successful search. calibrated_scores is set only when the
        # cross-encoder actually produced the scores above.
        if not results.calibrated_scores:
            raise SystemExit(
                "reranking did not run: scores fell back to RRF fusion, so "
                "the cross-encoder model was not loadable offline"
            )
        scores = await app.providers.rerank_async(
            "asynchronous reset naming", ["rst_n active-low reset", "unrelated text"]
        )
        if not (len(scores) == 2 and scores[0] > scores[1]):
            raise SystemExit(f"reranker did not produce a sane ranking: {scores}")
        print(f"  reranker scores: {scores[0]:.3e} > {scores[1]:.3e}")
    finally:
        app.close()


def main() -> int:
    print("corvidex-mcp offline verification")
    print(f"  python: {sys.version.split()[0]}  executable: {sys.executable}")
    _block_network()
    print("  outbound sockets: refused (every connect() raises)")
    print("models:")
    not_bundled = _report_models()
    if not_bundled:
        print(
            "warning: not every model is bundled; this installation is not "
            "the offline bundle, so the run below will only succeed if a "
            "populated embed-cache is present."
        )
    with tempfile.TemporaryDirectory(prefix="corvidex-verify-") as tmp:
        asyncio.run(_run(Path(tmp)))
    print("offline verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
