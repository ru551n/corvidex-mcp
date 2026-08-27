"""Retrieval-mode benchmark: quality and latency of the search strategies.

Drives the real production pipeline (real fastembed/ONNX model, real
chunkers, real sqlite-vec + FTS5 store, real search paths) over a local
repository and a query battery with ground truth, then reports per
search mode (``semantic`` / ``lexical`` / ``hybrid``):

- quality: top-1 and top-3 file hits against each query's expected
  file, strict top-1 (the unambiguous queries must hit top-1), and
  mean reciprocal rank (MRR);
- latency: per-query wall time (mode-specific: semantic/hybrid embed
  the query, lexical does not), mean over several runs.

Quality scoring mirrors the opt-in quality test
(tests/test_quality.py / docs/quality-testing.md): ``strict`` queries
count only as top-1 hits; every query also reports a top-3 hit.

Opt-in and manual: it loads the real embedding model and is NOT part
of the offline CI suite. Reuse a pre-provisioned model cache
(air-gapped pattern) via VHDL_RAG_EMBED_CACHE so it does not download.

Usage:
  VHDL_RAG_EMBED_CACHE=/path/to/fastembed-cache \
    uv run --no-sync python tools/bench_retrieval.py \
        --repo /path/to/tsfpga \
        --queries /path/to/queries.json \
        --runs 3 --out runs/bench_retrieval.json

queries.json: a JSON list of {"id", "query", "expect", "strict"} where
``expect`` is the repository-relative path of the file a correct search
must surface (top-1 for strict queries, top-3 otherwise).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

from vhdl_rag_mcp.config import AppConfig, EmbeddingsConfig, RepositoryConfig
from vhdl_rag_mcp.server import VhdlRagApp

logger = logging.getLogger("bench.retrieval")

MODES = ("semantic", "lexical", "hybrid")
DEFAULT_REPO = "/home/sebbe/vrmcp-data/repos/tsfpga"
DEFAULT_QUERIES = "/home/sebbe/lance-compare/queries.json"


def make_app(data_dir: Path, repo_name: str, repo_path: Path) -> VhdlRagApp:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Reuse a pre-provisioned model cache (air-gapped pattern) when
    # provided via VHDL_RAG_EMBED_CACHE, so the benchmark never downloads.
    cache = os.environ.get("VHDL_RAG_EMBED_CACHE")
    if cache and not (data_dir / "embed-cache").exists():
        (data_dir / "embed-cache").symlink_to(Path(cache))
    config = AppConfig(
        data_dir=data_dir,
        vhdl_ls_path="/nonexistent/vhdl_ls",  # deterministic structural chunking
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        embeddings=EmbeddingsConfig(),  # all defaults (small-en, batch 1)
        repositories=[RepositoryConfig(name=repo_name, path=repo_path)],
    )
    return VhdlRagApp(config)


def load_queries(path: Path) -> list[dict[str, Any]]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise SystemExit(f"queries file {path} must be a non-empty JSON list")
    for q in queries:
        for key in ("id", "query", "expect"):
            if key not in q:
                raise SystemExit(f"query {q.get('id', '?')!r} lacks {key!r}")
    return queries


def hit_rank(results: list, expect: str) -> int:
    """1-based rank of the first result whose file is ``expect`` (0 = miss)."""
    for rank, result in enumerate(results, start=1):
        if result.file == expect:
            return rank
    return 0


async def run(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries(Path(args.queries))
    data_dir = Path(args.data_dir)
    repo_name = Path(args.repo).name
    app = make_app(data_dir, repo_name, Path(args.repo))
    try:
        app.ensure_collections()
        app.migrate_index()
        check = app.selfcheck()
        if not check.required_ok:
            raise SystemExit(f"self-check failed: {check.summary()}")
        logger.info(
            "self-check: %s; indexing %s (%d queries)",
            check.summary(),
            repo_name,
            len(queries),
        )
        reports = await app.sync_all()
        for report in reports:
            if report.get("status") != "ok":
                raise SystemExit(f"indexing failed: {report}")

        results: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            latencies_ms: list[float] = []
            rows: list[dict[str, Any]] = []
            strict_top1 = strict_total = 0
            top1_hits = top3_hits = 0
            mrr = 0.0
            for query in queries:
                expect = query["expect"]
                times: list[float] = []
                hits: list[Any] = []
                for _ in range(args.runs):
                    start = time.perf_counter()
                    found = app.retrieval.search_knowledge(
                        query["query"], limit=args.limit, mode=mode
                    )
                    times.append((time.perf_counter() - start) * 1000.0)
                    hits.append(found)
                rank = hit_rank(hits[-1], expect)
                top1 = rank == 1
                top3 = 1 <= rank <= 3
                top1_hits += top1
                top3_hits += top3
                if query.get("strict"):
                    strict_total += 1
                    strict_top1 += top1
                mrr += 1.0 / rank if rank else 0.0
                latencies_ms.extend(times)
                rows.append(
                    {
                        "id": query["id"],
                        "expect": expect,
                        "strict": bool(query.get("strict")),
                        "rank": rank,
                        "top1": top1,
                        "top3": top3,
                        "top_files": [r.file for r in hits[-1][:3]],
                    }
                )
            results[mode] = {
                "queries": rows,
                "strict_top1": strict_top1,
                "strict_total": strict_total,
                "top1": top1_hits,
                "top3": top3_hits,
                "mrr": round(mrr / len(queries), 4),
                "latency_ms_mean": round(statistics.fmean(latencies_ms), 3),
                "latency_ms_p50": round(statistics.median(latencies_ms), 3),
                "latency_ms_max": round(max(latencies_ms), 3),
                "runs": args.runs,
            }
        return {
            "repo": args.repo,
            "queries": args.queries,
            "limit": args.limit,
            "chunks": app.store.count(),
            "self_check": check.summary(),
            "modes": results,
        }
    finally:
        app.close()


def render(result: dict[str, Any]) -> str:
    """Human-readable report (markdown)."""
    lines = [
        f"# Retrieval-mode benchmark: {result['repo']}",
        "",
        f"- chunks indexed: {result['chunks']}",
        f"- self-check: {result['self_check']}",
        f"- query limit: {result['limit']}, runs: {result['modes'][MODES[0]]['runs']}",
        "",
        "| mode | strict top-1 | top-1 | top-3 | MRR | mean ms | p50 ms | max ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        r = result["modes"][mode]
        lines.append(
            f"| {mode} "
            f"| {r['strict_top1']}/{r['strict_total']} "
            f"| {r['top1']}/{len(r['queries'])} "
            f"| {r['top3']}/{len(r['queries'])} "
            f"| {r['mrr']:.4f} "
            f"| {r['latency_ms_mean']} "
            f"| {r['latency_ms_p50']} "
            f"| {r['latency_ms_max']} |"
        )
    lines.append("")
    for mode in MODES:
        r = result["modes"][mode]
        lines.append(f"## {mode}")
        lines.append("")
        for row in r["queries"]:
            mark = "TOP1" if row["top1"] else ("TOP3" if row["top3"] else "MISS")
            strict = " [strict]" if row["strict"] else ""
            lines.append(
                f"- [{mark}] {row['id']}{strict} expect={row['expect']} "
                f"rank={row['rank'] or '-'} top3={row['top_files']}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="local repository path")
    parser.add_argument(
        "--queries", default=DEFAULT_QUERIES, help="queries.json ground-truth file"
    )
    parser.add_argument("--runs", type=int, default=3, help="timed runs per query")
    parser.add_argument("--limit", type=int, default=8, help="search result limit")
    parser.add_argument(
        "--data-dir",
        default="/tmp/vhdl-rag-bench-retrieval",
        help="scratch data dir (fresh index per run)",
    )
    parser.add_argument("--out", default="runs/bench_retrieval.json")
    args = parser.parse_args(argv)

    # A fresh index per run: the benchmark measures steady-state search,
    # and a leftover index from another repo/corpus would pollute it.
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = asyncio.run(run(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(render(result))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
