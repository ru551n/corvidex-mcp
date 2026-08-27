# Qdrant vs LanceDB vs SQLite+sqlite-vec — real-world three-way backend comparison

Branch `explore/sqlite-vec` replaces the embedded Qdrant vector store
with SQLite + sqlite-vec behind the same public API. This document
records the head-to-head measurement of the two embedded replacement
backends against the incumbent one on real-world HDL repositories,
covering **indexing speed**, **resource usage**, **retrieval quality**,
and **query latency**. The two-way Qdrant vs LanceDB measurement
(alone) is recorded in `docs/lancedb-comparison.md` on
`explore/lancedb`.

## Method

- **Corpus** (two real-world repositories, indexed as local working
  repos through the full production pipeline — git sync, routing,
  structural chunking, dense embedding):
  - `tsfpga` — FPGA flow framework (VHDL + Python), 114 files
  - `hdl-modules` — reusable VHDL IP library, 239 files
  - 1,646 chunks total: 288 hdl, 432 docs, 926 code
- **Backends** — all embedded in the server process, all persistent:
  - `main` @ `ce78307`: Qdrant local mode (`QdrantClient(path=…)`),
    hybrid dense + BM25 sparse
  - `explore/lancedb` @ `7601e6e`: LanceDB (3 Lance tables), hybrid
    dense + FTS (simple tokenizer, stop words removed), RRF fusion —
    the BM25 model is never loaded and no sparse vectors exist (the
    branch's later commits are dead-code/docs cleanup only and do not
    touch the measured path)
  - `explore/sqlite-vec` @ `d04a2e1`: one SQLite file (WAL mode) per
    data dir; per collection a payload table + `vec0` vector table +
    FTS5 table joined by rowid; unit-normalized dense vectors searched
    by flat exact KNN (SIMD), FTS5 keyword ranking (BM25 with
    `tokenchars '_'` so VHDL identifiers stay whole tokens), RRF K=60
    fusion — the BM25 model is never loaded and no sparse vectors exist
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
- **Store size** — backend payload on disk after clean close, excluding
  the shared dense-embedding output cache (~3.0 MB, byte-identical
  across the three backends) and the app state file.

## Results

| Metric (lower is better unless noted) | Qdrant (`main`) | LanceDB (`explore/lancedb`) | SQLite (`explore/sqlite-vec`) |
|---|---|---|---|
| Index wall time, 2 rounds | 53.3 s / 53.1 s | 35.5 s / 36.7 s | 36.5 s / 34.8 s |
| Peak RSS (indexing) | **663 MB** | 888 MB | 716 MB |
| Store payload on disk | 12.7 MB (dense + sparse) | **4.9 MB** (columnar, dense + FTS) | 12.1 MB (payload + vec0 + FTS5, one file) |
| Index quality, top-3 hits | 7/16 (strict top-1 0/11) | 7/16 (strict top-1 1/11) | 7/16 (strict top-1 1/11) |
| Query latency, median of 16 medians | 33.0 ms (range 28.3–60.0) | 24.7 ms (range 22.2–28.6) | **9.4 ms (range 8.0–11.9)** |

Setup (model load + table creation) was 0.13–0.20 s (Qdrant),
0.10–0.34 s (LanceDB), and 0.10–0.31 s (SQLite) — negligible either
way. All three backends indexed the identical 1,646 chunks.

Per-query quality (rank of the expected file in the top-3; — = miss):

| Query (expected file) | Qdrant | LanceDB | SQLite |
|---|---|---|---|
| `tsfpga-vivado-project` (`tsfpga/vivado/project.py`) | — | — | — |
| `tsfpga-timing-parser` (`tsfpga/vivado/timing_parser.py`) | 3 | 3 | 3 |
| `tsfpga-simlib-ghdl` (`tsfpga/vivado/simlib_ghdl.py`) | — | — | 3 |
| `tsfpga-utilization` (`tsfpga/vivado/hierarchical_utilization_parser.py`) | 3 | 3 | 3 |
| `tsfpga-xdc-constraints` (`tsfpga/constraint.py`) | 2 | 2 | 2 |
| `tsfpga-git-simulation` (`tsfpga/git_simulation_subset.py`) | 3 | 3 | 3 |
| `tsfpga-tcl-hook` (`tsfpga/build_step_tcl_hook.py`) | — | — | — |
| `tsfpga-version-bump` (`tsfpga/tools/version_number_handler.py`) | 3 | — | — |
| `hdl-axi-to-lite` (`modules/axi_lite/src/axi_to_axi_lite.vhd`) | — | — | **1** |
| `hdl-async-fifo` (`modules/fifo/src/asynchronous_fifo.vhd`) | — | — | — |
| `hdl-resync-level` (`modules/resync/src/resync_level.vhd`) | — | — | — |
| `hdl-sine-lookup` (`modules/sine_generator/src/sine_lookup.vhd`) | — | 3 | — |
| `hdl-debounce` (`modules/common/src/debounce.vhd`) | 2 | **1** | 2 |
| `hdl-unsigned-divider` (`modules/math/src/unsigned_divider.vhd`) | — | — | — |
| `hdl-lfsr-fibonacci` (`modules/lfsr/src/lfsr_fibonacci_single.vhd`) | — | — | — |
| `hdl-handshake-pipeline` (`modules/common/src/handshake_pipeline.vhd`) | 2 | 2 | — |

## Interpretation

**Indexing speed.** LanceDB and SQLite are effectively tied (35.9 s vs
35.4 s average over two rounds), each ~33 % faster than Qdrant (53.0
s). Indexing time is dominated by the shared dense-embedding step;
Qdrant's surplus is the fixed cost of its sparse leg — embedding the
1,646 chunks with the BM25 model on top of the dense ones. The FTS-based
backends never load the BM25 model.

**Query latency.** SQLite is 3.5× faster than Qdrant (9.4 vs 33.0 ms)
and 2.6× faster than LanceDB (9.4 vs 24.7 ms), with the tightest
spread (8.0–11.9 ms). Qdrant pays for a second (BM25) query embedding
plus its ANN + sparse fusion path; LanceDB's hybrid legs run through
the pyarrow/lance query machinery; SQLite's legs are a raw vec0 KNN
scan and an FTS5 ranked query, fused in Python. The shared dense query
embedding is a constant floor for all three.

**Resources.** Peak RSS during indexing: Qdrant 663 MB < SQLite
716 MB < LanceDB 888 MB (all include the ONNX embedding runtime and
the in-process corpus). On disk, LanceDB's columnar payload is the
smallest (4.9 MB); Qdrant (12.7 MB, dense + sparse) and SQLite (12.1
MB: payload JSON + float32 vec0 vectors + FTS5 inverted index in one
file) are comparable. SQLite's single-file store is also the simplest
to back up and inspect with standard `sqlite3` tooling.

**Quality.** A three-way tie at this scale: 7/16 top-3 hits across the
board, and the six queries all three backends *miss* are the same six.
The remaining spread: Qdrant's unique hit is `tsfpga-version-bump`
(#3), LanceDB's is `hdl-sine-lookup` (#3), and SQLite's are
`tsfpga-simlib-ghdl` (#3) and `hdl-axi-to-lite` (top-1 — the only
strict top-1 besides LanceDB's `hdl-debounce`). SQLite's one miss that
the others hit is `hdl-handshake-pipeline`, where the testbench that
instantiates the pipeline (`tb_handshake_bfm.vhd`) ranks #2 — the
`handshake_pipeline` identifier appears just as prominently there, a
corpus-level effect. SQLite's identifier wins come from its FTS5
tokenizer (`tokenchars '_'`): `axi_to_axi_lite` and `simlib_ghdl` stay
whole tokens, giving exact-identifier ranking where the simple-tokenizer
(LanceDB) and BM25-sparse (Qdrant) legs split on `_` and dilute the
signal. Where the backends *miss*, the cause is corpus-level, not
backend-level: keyword-heavy release notes (`.rst`) outrank
implementation files for paraphrased queries, and testbenches compete
with `src/` files. The retrieval stack (RRF cross-collection fusion) is
shared code, identical on all three branches, so backend-specific
differences surface only in the per-collection legs.

**Scale caveat.** 1,646 vectors is below the threshold where ANN index
structures matter — all three backends effectively search
exhaustively here, so the latency gap is dominated by the shared
embedding path and engine overhead, not index structure. The scaling
shapes do differ: sqlite-vec's flat exact KNN degrades linearly with
corpus size (no ANN index in 0.1.9), while Qdrant (HNSW) and LanceDB
(IVF-PQ) offer approximate indexes for large corpora. Re-run this
harness at a larger corpus before drawing conclusions about query
latency at scale; the indexing-speed and RAM gaps should persist.

**Bottom line.** At this corpus scale all three backends are
production-viable with statistically identical retrieval quality.
Differentiators: SQLite has the lowest query latency (9.4 ms) and the
simplest single-file store; LanceDB has the smallest on-disk payload
(4.9 MB); Qdrant has the leanest peak RAM (663 MB).

## Reproducing

```sh
git worktree add /tmp/wt-main main
git worktree add --detach /tmp/wt-lancedb <explore/lancedb-tip>
git worktree add --detach /tmp/wt-sqlite <explore/sqlite-vec-tip>
(cd /tmp/wt-main    && uv sync && \
    uv run --no-sync python /path/to/harness.py --label qdrant  --rounds 2)
(cd /tmp/wt-lancedb && uv sync && \
    uv run --no-sync python /path/to/harness.py --label lancedb --rounds 2)
(cd /tmp/wt-sqlite  && uv sync && \
    uv run --no-sync python /path/to/harness.py --label sqlite  --rounds 2)
```

`harness.py` (kept in the comparison sandbox, not in the repo) writes
`runs/<label>/results.json` with per-round timings/RSS/counts and
per-query top-3/latency detail.