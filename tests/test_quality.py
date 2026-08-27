"""End-to-end RAG quality tests (real embedding models).

Indexes a small fixture repository (``tests/quality/corpus/``) through
the full production pipeline — git sync, chunking, dense
embedding, hybrid (dense + full-text) search — with the **real** fastembed models
(the defaults from the ``[embeddings]`` config), and asserts that a
fixed query battery (``tests/quality/queries.json``) retrieves the
expected source files.

Opt-in: skipped unless ``VHDL_RAG_RUN_QUALITY=1`` is set. The CI job
``quality-e2e`` (ubuntu-latest only) sets it; the rest of the matrix
stays offline and fast. Local run (downloads ~1.5 GB of models once):

    VHDL_RAG_RUN_QUALITY=1 uv run pytest tests/test_quality.py -v

Notes:

- ``vhdl_ls``/``veridian`` are deliberately not used: the pipeline
  falls back to structural chunking when the binaries are absent, and
  the quality assertions (file-level, dense + full-text legs) do not
  depend on LSP symbols. This keeps CI and local runs identical.
- Embedding determinism is asserted separately: the same text must
  embed to bit-identical vectors (guards against non-deterministic
  model or ORT upgrades).

Thresholds are tuned against this fixture (see
``docs/quality-testing.md``); re-validate before changing them.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from vhdl_rag_mcp.config import AppConfig, EmbeddingsConfig, RepositoryConfig
from vhdl_rag_mcp.embeddings.providers import EmbeddingProviders
from vhdl_rag_mcp.models import CollectionName
from vhdl_rag_mcp.server import VhdlRagApp

RUN_QUALITY = os.environ.get("VHDL_RAG_RUN_QUALITY") == "1"

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "quality" / "corpus"
QUERIES = json.loads((HERE / "quality" / "queries.json").read_text(encoding="utf-8"))

#: Minimum top-3 file hits across the whole battery.
MIN_TOP3_HITS = 10
#: Every "strict" query (unambiguous, high-confidence) must hit top-1.
STRICT_MUST_TOP1 = True

pytestmark = pytest.mark.skipif(
    not RUN_QUALITY,
    reason="opt-in quality test: set VHDL_RAG_RUN_QUALITY=1 (docs/quality-testing.md)",
)

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "quality",
    "GIT_AUTHOR_EMAIL": "quality@example.com",
    "GIT_COMMITTER_NAME": "quality",
    "GIT_COMMITTER_EMAIL": "quality@example.com",
}


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=GIT_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def quality_data_dir() -> Path:
    """Base for vector store + state; embed-cache lives under it.

    CI points ``VHDL_RAG_QUALITY_DATA`` at the actions/cache path so
    model downloads happen once; locally this is a temp dir.
    """
    env = os.environ.get("VHDL_RAG_QUALITY_DATA")
    return (
        Path(env)
        if env
        else Path(os.environ.get("TMPDIR", "/tmp")) / "vhdl-rag-quality-data"
    )


@pytest.fixture
def quality_remote(tmp_path: Path) -> Path:
    """The fixture corpus as a local git remote (file:// sync path)."""
    remote = tmp_path / "remote"
    remote.mkdir()
    for src in sorted(CORPUS_DIR.rglob("*")):
        if src.is_file():
            dst = remote / src.relative_to(CORPUS_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    git(remote, "init", "-q", "-b", "main")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "fixture corpus")
    return remote


def make_config(data_dir: Path, remote: Path) -> AppConfig:
    # Non-existent LSP/veridian paths -> deterministic structural
    # chunking fallback (see module docstring).
    return AppConfig(
        data_dir=data_dir,
        sync_interval=3600,
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        embeddings=EmbeddingsConfig(
            dense_threads=2,  # CI runners are small; keep RAM/cores low
            dense_enable_cpu_mem_arena=False,
        ),
        repositories=[RepositoryConfig(name="quality", url=str(remote), ref="main")],
    )


def quality_app(data_dir: Path, remote: Path) -> VhdlRagApp:
    config = make_config(data_dir, remote)
    app = VhdlRagApp(config, providers=EmbeddingProviders(config))
    app.ensure_collections()
    return app


async def test_quality_battery(quality_remote: Path) -> None:
    data_dir = quality_data_dir() / "battery"
    data_dir.mkdir(parents=True, exist_ok=True)
    app = quality_app(data_dir, quality_remote)
    try:
        t0 = time.monotonic()
        await app.sync_all()
        sync_s = time.monotonic() - t0
        n_hdl = app.store.count(CollectionName.HDL)
        n_docs = app.store.count(CollectionName.DOCS)
        n_code = app.store.count(CollectionName.CODE)
        assert n_hdl >= 3 and n_docs >= 3 and n_code >= 2, (
            "fixture did not index as expected: "
            f"hdl={n_hdl} docs={n_docs} code={n_code}"
        )

        lines: list[str] = []
        top3_hits = 0
        strict_top1 = 0
        strict_total = 0
        for q in QUERIES:
            if q["collection"] == "knowledge":
                results = app.retrieval.search_knowledge(q["query"], limit=5)
            else:
                results = app.retrieval.search(
                    CollectionName(q["collection"]), q["query"], limit=5
                )
            top_files = [r.file for r in results[:3]]
            hit3 = any(q["expect"] == f for f in top_files)
            hit1 = bool(results) and results[0].file == q["expect"]
            top3_hits += hit3
            if q["strict"]:
                strict_total += 1
                strict_top1 += hit1
            mark = "OK " if hit1 else ("ok3" if hit3 else "MISS")
            lines.append(
                f"  [{mark}] {q['id']:<22} expect={q['expect']:<24} top3={top_files}"
            )

        report = "\n".join(
            [
                "",
                "quality battery (real models, structural chunking):",
                f"  sync={sync_s:.1f}s chunks: hdl={n_hdl} docs={n_docs} code={n_code}",
                *lines,
                f"  top-3 hits: {top3_hits}/{len(QUERIES)} (min {MIN_TOP3_HITS})",
                f"  strict top-1: {strict_top1}/{strict_total}",
                "",
            ]
        )
        print(report)

        assert top3_hits >= MIN_TOP3_HITS, report
        if STRICT_MUST_TOP1:
            assert strict_top1 == strict_total, report
    finally:
        app.close()


def test_quality_embedding_determinism() -> None:
    """Same text -> bit-identical vectors across repeated calls."""
    data_dir = quality_data_dir() / "determinism"
    data_dir.mkdir(parents=True, exist_ok=True)
    config = AppConfig(data_dir=data_dir, repositories=[])
    providers = EmbeddingProviders(config)
    coll = CollectionName.HDL
    dim = providers.dimension(coll)
    # The default model (jina v2 small-en) is 512-dim.
    assert dim == 512

    texts = [
        "entity sample is port (clk : in std_logic; q : out std_logic)"
        " end entity sample;",
        "pip install my-toolchain",
        "run the synthesis script to generate the bitstream",
    ]
    first = providers.embed_passages(coll, texts)
    second = providers.embed_passages(coll, texts)
    assert first == second, "dense embedding is not deterministic across calls"
    assert all(len(v) == dim for v in first)
