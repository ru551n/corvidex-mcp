# LanceDB vs Qdrant — real-world backend comparison

Branch `explore/lancedb` replaces the embedded Qdrant vector store with
embedded LanceDB behind the same public API. This document records the
head-to-head measurement of the two backends on real-world HDL
repositories, covering **indexing speed**, **resource usage**,
**retrieval quality**, and **query latency**.

## Method

- **Corpus** (two real-world repositories, indexed as local working
  repos through the full production pipeline — git sync, routing,
  structural chunking, dense embedding):
  - `tsfpga` — FPGA flow framework (VHDL + Python), 114 files
  - `hdl-modules` — reusable VHDL IP library, 239 files
  - 1,646 chunks total: 288 hdl, 432 docs, 926 code
- **Backends** — both embedded in the server process, both persistent:
  - `main` @ `ce78307`: Qdrant local mode (`QdrantClient(path=…)`),
    hybrid dense + BM25 sparse
  - `explore/lancedb` @ `7601e6e`: LanceDB (3 Lance tables), hybrid
    dense + FTS (simple tokenizer, stop words removed), RRF fusion —
    the BM25 model is never loaded and no sparse vectors exist
- **Identical harness** (`/home/sebbe/lance-compare/harness.py`) run in
  each branch's own venv: fresh data dir per round, 2 cold-indexing
  rounds per backend, then the quality/latency battery on the final
  index. Same machine (32-core, 60 GB RAM), same default jina v2
  small-en model (512-dim), 4 ONNX threads, 1 indexing worker, CPU
  memory arena off, process contained with `ulimit -v 24 GiB`.
- **Quality battery** — 16 user-style queries (8 per repo: semantic
  paraphrases plus raw identifiers like `build_step_tcl_hook` and
  `handshake_pipeline`), issued through the user-facing
  `search_knowledge` (all collections, limit 5, top-3 counted), with
  hand-verified expected files.
- **Latency** — per query, 5 runs, median reported (includes query
  embedding; Qdrant's BM25 query embedding included in its time).

## Results

| Metric (lower is better unless noted) | Qdrant (`main`) | LanceDB (`explore/lancedb`) |
|---|---|---|
| Index wall time, 2 rounds | 51.5 s / 50.6 s | **34.4 s / 35.0 s** |
| Peak RSS (indexing) | **660 MB** | 881–884 MB |
| Store payload on disk | 12.7 MB (dense + sparse) | **5.0 MB** (columnar, dense + FTS) |
| Index quality, top-3 hits | 7/16 (strict top-1 0/11) | 7/16 (strict top-1 1/11) |
| Query latency, median of 16 medians | 32.7 ms (range 26.0–54.7) | **24.1 ms (range 23.0–26.8)** |

Setup (model load + table creation) was 0.16–0.21 s (Qdrant) and
0.11–0.33 s (LanceDB) — negligible either way. Both backends indexed
the identical 1,646 chunks.

## Interpretation

**Speed.** LanceDB indexes ~32 % faster on this corpus and answers
queries ~26 % faster with much tighter latency (no BM25 CPU leg,
native RRF). The Qdrant numbers include embedding 1,646 chunks with
the BM25 model and, per query, embedding the query twice — a fixed
cost of its sparse leg that LanceDB's FTS leg avoids entirely.

**Resources.** Qdrant's embedded store is leaner in RAM (660 vs
~882 MB peak; the difference is pyarrow/lance runtime plus ONNX
buffering in the LanceDB process) and the on-disk payload is ~2.5×
larger because it stores sparse vectors alongside the dense ones.
LanceDB writes half as much and reads it back from a columnar layout.

**Quality.** Effectively tied at this scale: 8 of 16 queries return
the identical top-3 on both backends; the two disagree on exactly one
hit each (`tsfpga-version-bump` for Qdrant, `hdl-sine-lookup` for
LanceDB). LanceDB scores the only top-1. Where both backends *miss*,
the cause is corpus-level, not backend-level: keyword-heavy release
notes (`.rst`) outrank implementation files for paraphrased queries,
and testbenches compete with `src/` files. The retrieval stack (RRF
cross-collection fusion) is shared code and identical on both
branches, so backend-specific differences surface only in the
per-collection legs.

**Scale caveat.** 1,646 vectors is below the threshold where ANN
index structures (HNSW vs IVF-PQ) matter — both backends effectively
search exhaustively here, so the query-latency gap is dominated by
the embedding path, not the index. Re-run this harness at a larger
corpus before drawing conclusions about query latency at scale; the
indexing-speed and RSS gaps should persist.

## Reproducing

```sh
git worktree add /tmp/wt-main main
git worktree add --detach /tmp/wt-lancedb <explore/lancedb-tip>
(cd /tmp/wt-main    && uv sync && \
    uv run --no-sync python /path/to/harness.py --label qdrant  --rounds 2)
(cd /tmp/wt-lancedb && uv sync && \
    uv run --no-sync python /path/to/harness.py --label lancedb --rounds 2)
```

`harness.py` (kept in the comparison sandbox, not in the repo) writes
`runs/<label>/results.json` with per-round timings/RSS/counts and
per-query top-3/latency detail.