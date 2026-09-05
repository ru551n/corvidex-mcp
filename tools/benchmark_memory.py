"""Memory-decomposition and storage benchmark for the SQLite/sqlite-vec backend.

Purpose (§31/§32): identify *where* peak RSS and wall time come from during
indexing, so we do not assume the database backend causes memory growth. It
drives the real production pipeline (real fastembed/ONNX model, real
chunkers, real sqlite-vec + FTS5 store) over local working repositories and
reports, per stage:

  A  chunking only            (parse files -> Chunk objects)
  B  chunking + tokenization  (tokenize every chunk's content)
  C  dense embedding, no DB   (embed passages; discard vectors)
  D  DB writes, precomputed   (upsert payload + vec0 + FTS5 with fixed vectors)
  F  full pipeline            (cold sync_all, the real end-to-end number)

plus an embedding input-length sweep (512..8192 tokens) to confirm that
sequence length / ONNX inference is the primary memory driver.

Opt-in and manual: it loads the real embedding model (downloaded once to the
embed cache) and is intentionally NOT part of the offline CI suite.

Usage:
  uv run --no-sync python tools/benchmark_memory.py \
      --repo tsfpga=/path/to/tsfpga --repo hdl=/path/to/hdl-modules \
      --out runs/bench_memory.json
  uv run --no-sync python tools/benchmark_memory.py --sweep --out runs/sweep.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import resource
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from corvidex_mcp.config import AppConfig, EmbeddingsConfig, RepositoryConfig
from corvidex_mcp.indexing.code import chunk_code_file
from corvidex_mcp.indexing.docs import chunk_doc_file
from corvidex_mcp.indexing.vhdl import chunk_vhdl_file
from corvidex_mcp.models import Chunk, CollectionName
from corvidex_mcp.routing import classify_file
from corvidex_mcp.server import VhdlRagApp

logger = logging.getLogger("bench.memory")

SWEEP_LENGTHS = (512, 1024, 2048, 3000, 4096, 8192)


def peak_rss_mb() -> float:
    """Process high-water-mark RSS in MB (Linux: KB, macOS: bytes)."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if sys.platform != "darwin" else kb / 1024.0 / 1024.0


def make_app(data_dir: Path, repos: list[tuple[str, Path]]) -> VhdlRagApp:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Reuse a pre-provisioned model cache (air-gapped pattern) when provided
    # via CORVIDEX_EMBED_CACHE, so the benchmark does not re-download the model.
    cache = os.environ.get("CORVIDEX_EMBED_CACHE")
    if cache and not (data_dir / "embed-cache").exists():
        (data_dir / "embed-cache").symlink_to(Path(cache))
    config = AppConfig(
        data_dir=data_dir,
        vhdl_ls_path="/nonexistent/vhdl_ls",  # deterministic structural chunking
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        embeddings=EmbeddingsConfig(),  # all defaults (small-en, 4 threads, 1 worker)
        repositories=[RepositoryConfig(name=name, path=path) for name, path in repos],
    )
    return VhdlRagApp(config)


def list_files(repo: Path) -> list[str]:
    """Repository-relative paths of all regular files (skipping .git)."""
    files: list[str] = []
    for root, dirs, names in os.walk(repo):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in names:
            rel = Path(root).relative_to(repo) / name
            files.append(rel.as_posix())
    return sorted(files)


def chunk_repository(app: VhdlRagApp, name: str, commit: str) -> list[Chunk]:
    """Structurally chunk every indexable file in one local working repo."""
    cfg = app.config.repository(name)
    repo_dir = app.git.repo_dir(cfg)
    chunks: list[Chunk] = []
    for rel in list_files(repo_dir):
        kind = classify_file(rel, cfg.enabled_collections, cfg.exclude)
        if kind is None:
            continue
        content = app.git.read_file(cfg, rel)
        if kind.collection is CollectionName.HDL and kind.language == "vhdl":
            chunks.extend(chunk_vhdl_file(cfg, rel, content, commit, lsp_symbols=None))
        elif kind.collection is CollectionName.DOCS:
            chunks.extend(chunk_doc_file(cfg, rel, content, commit, kind.language))
        else:
            # code collection, and HDL (Verilog/SV) files via the generic
            # parser — the benchmark profiles the pipeline shape, not per-HDL
            # analyzer fidelity.
            chunks.extend(chunk_code_file(cfg, rel, content, commit, kind.language))
    return chunks


def token_counter(app: VhdlRagApp):
    """Return a callable token->int using the already-loaded model's tokenizer.

    Reuses the single in-process model (no second model load, so the measured
    RSS is the real single-model footprint). Returns ``None`` when the model
    exposes no tokenizer (e.g. test fakes); token counts are then omitted.
    """
    provider = app.providers._dense_provider(CollectionName.HDL)
    tokenizer = provider._tokenizer()
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None

    def count(text: str) -> int:
        # Count the raw chunk length: the model's tokenizer carries the
        # dense_max_tokens truncation cap, which would undercount long chunks.
        truncated = False
        if hasattr(tokenizer, "no_truncation"):
            tokenizer.no_truncation()
            truncated = True
        try:
            return len(list(tokenizer.encode(text).ids))
        finally:
            if truncated:
                tokenizer.enable_truncation(
                    max_length=app.config.embeddings.dense_max_tokens
                )

    return count


async def run_stages(repos: list[tuple[str, Path]], out: Path) -> None:
    data_dir = Path(tempfile.mkdtemp(prefix="corvidex-bench-"))
    try:
        app = make_app(data_dir, repos)
        counts_by_repo: dict[str, int] = {}
        chunks: list[Chunk] = []
        first_commit = "0" * 40

        t0 = time.perf_counter()
        for name, _ in repos:
            got = chunk_repository(app, name, first_commit)
            counts_by_repo[name] = len(got)
            chunks.extend(got)
        stage_a_s = time.perf_counter() - t0
        a = {
            "stage": "A_chunking",
            "wall_s": round(stage_a_s, 2),
            "chunks": len(chunks),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        logger.info("A: %s", a)

        app.ensure_collections()  # loads the model + creates tables

        count_tokens = token_counter(app)
        t0 = time.perf_counter()
        tokens = 0
        if count_tokens is not None:
            tokens = sum(count_tokens(c.content) for c in chunks)
        stage_b_s = time.perf_counter() - t0
        b = {
            "stage": "B_chunk+tokenize",
            "wall_s": round(stage_b_s, 2),
            "tokens": tokens,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        logger.info("B: %s", b)

        # C: dense embedding with no DB writes.
        by_collection: dict[CollectionName, list[Chunk]] = {}
        for c in chunks:
            by_collection.setdefault(c.collection, []).append(c)
        t0 = time.perf_counter()
        for coll, items in by_collection.items():
            app.providers.embed_passages(coll, [c.content for c in items])
        stage_c_s = time.perf_counter() - t0
        c = {
            "stage": "C_embed_no_write",
            "wall_s": round(stage_c_s, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        logger.info("C: %s", c)

        # D: DB writes with precomputed (freshly computed) vectors.
        t0 = time.perf_counter()
        for coll, items in by_collection.items():
            dense = app.providers.embed_passages(coll, [x.content for x in items])
            app.store.upsert_chunks(items, dense)
        stage_d_s = time.perf_counter() - t0
        store_bytes = app.config.sqlite_index_path.stat().st_size
        d = {
            "stage": "D_db_writes_precomputed",
            "wall_s": round(stage_d_s, 2),
            "index_bytes": store_bytes,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        logger.info("D: %s", d)

        # F: full pipeline on a fresh app/data dir (the real end-to-end number).
        data_dir2 = Path(tempfile.mkdtemp(prefix="corvidex-bench-full-"))
        app2 = make_app(data_dir2, repos)
        t_setup = time.perf_counter()
        app2.ensure_collections()  # mirrors production startup before sync
        setup_s = time.perf_counter() - t_setup
        t0 = time.perf_counter()
        await app2.sync_all()
        stage_f_s = time.perf_counter() - t0
        f = {
            "stage": "F_full_pipeline",
            "setup_s": round(setup_s, 2),
            "wall_s": round(stage_f_s, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        logger.info("F: %s", f)

        result = {
            "repos": {n: str(p) for n, p in repos},
            "chunks_by_repo": counts_by_repo,
            "total_chunks": len(chunks),
            "stages": [a, b, c, d, f],
            "notes": (
                "peak_rss_mb is the process high-water mark (cumulative); "
                "a stage's *added* peak is the delta vs the previous stage. "
                "D measures payload+vec0+FTS5 together (same transaction). "
                "F runs a fresh app/data dir."
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        app.close()
        app2.close()
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(data_dir2, ignore_errors=True)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _sweep_text(tokenizer: Any, n_tokens: int) -> str:
    """Text of exactly ``n_tokens`` model tokens (tokenizer-defined).

    Built by decoding a tiled pool of real VHDL-ish token ids so the
    tokenizer (not byte length) defines the true input length.
    """
    unit = (
        "process p_control is begin if rising_edge(clk) then "
        "if reset_n = '0' then state <= IDLE; elsif enable = '1' then "
        "state <= NEXT_STATE; end if; data_out <= data_in + 1; end if; end process"
    )
    pool = list(tokenizer.encode(unit, is_pretokenized=False).ids)
    if not pool:
        raise RuntimeError("tokenizer produced no ids for the sweep unit")
    ids = pool * (n_tokens // len(pool) + 1)
    return tokenizer.decode(ids[:n_tokens])


async def run_sweep(out: Path) -> None:
    data_dir = Path(tempfile.mkdtemp(prefix="corvidex-bench-sweep-"))
    try:
        app = make_app(data_dir, [])
        app.ensure_collections()  # loads the model
        provider = app.providers._dense_provider(CollectionName.HDL)
        tokenizer = provider._tokenizer()
        if tokenizer is None:
            raise SystemExit("sweep requires the model's tokenizer (no test fakes)")
        # The provider configures tokenizer truncation to dense_max_tokens;
        # the sweep must measure the raw model at each requested length.
        if hasattr(tokenizer, "no_truncation"):
            tokenizer.no_truncation()
        dense_model = provider._dense  # raw model: no truncation, no cache
        if dense_model is None:
            raise SystemExit("dense model not loaded")
        rows = []
        for n in SWEEP_LENGTHS:
            text = _sweep_text(tokenizer, n)
            actual_tokens = len(list(tokenizer.encode(text).ids))
            t0 = time.perf_counter()
            # passage_embed is a lazy generator: consume it to force inference.
            vectors = list(dense_model.passage_embed([text], batch_size=1))
            assert len(vectors) == 1
            wall = time.perf_counter() - t0
            row = {
                "requested_tokens": n,
                "actual_tokens": actual_tokens,
                "wall_s": round(wall, 3),
                "peak_rss_mb": round(peak_rss_mb(), 1),
            }
            rows.append(row)
            logger.info("sweep %s", row)
        if hasattr(tokenizer, "enable_truncation"):
            tokenizer.enable_truncation(
                max_length=app.config.embeddings.dense_max_tokens
            )
        result = {
            "lengths": SWEEP_LENGTHS,
            "rows": rows,
            "median_wall_s": round(statistics.median(r["wall_s"] for r in rows), 3),
            "note": (
                "peak_rss_mb is the cumulative process high-water mark; each "
                "point embeds one passage of the given (tokenized) length via "
                "the raw ONNX model, bypassing the provider's token cap and "
                "the dense-vector cache. Model is loaded once before the sweep."
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        app.close()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _parse_repos(specs: list[str]) -> list[tuple[str, Path]]:
    repos: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--repo expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        repos.append((name, Path(path).expanduser()))
    return repos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="local working repository (repeatable)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run only the embedding input-length sweep"
    )
    parser.add_argument(
        "--out",
        default="runs/bench_memory.json",
        metavar="PATH",
        help="where to write the JSON report",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.sweep:
        asyncio.run(run_sweep(Path(args.out)))
    else:
        repos = _parse_repos(args.repo)
        if not repos:
            raise SystemExit("provide at least one --repo NAME=PATH (or --sweep)")
        asyncio.run(run_stages(repos, Path(args.out)))


if __name__ == "__main__":
    main()
