# Quality testing (CI)

The unit/e2e suite runs entirely offline with fake embedding providers
and a faked/disabled reranker (see `tests/conftest.py`). That verifies
*plumbing*, not *retrieval quality*. The quality job (`quality-e2e` in
`.github/workflows/ci.yml`) closes that gap: it runs the **real**
production pipeline — git sync, chunking, dense embedding, hybrid
(dense + full-text, RRF) search, query expansion, and cross-encoder
reranking — on a small fixture corpus with the **real** fastembed
models, and asserts that a fixed query battery retrieves the expected
source files.

## What runs where

| Job / env | OS | Models | Purpose |
|---|---|---|---|
| `test` matrix | ubuntu / windows / macos / RHEL / arm64 | fakes | plumbing, fast, offline |
| `quality-e2e` | **ubuntu-latest only** | real (jina v2 small-en for hdl/docs, jina v2 base-code for code + FTS5 + MiniLM-L-6 reranker) | end-to-end retrieval quality |
| `tests/test_quality.py` locally | any | real | same gate, on demand |

`tests/test_quality.py` is skipped unless
`CORVIDEX_RUN_QUALITY=1` is set, so it never runs inside the matrix
jobs (which do not set the variable) and never downloads models on
Windows/macOS/RHEL/arm64 runners.

## What the tests assert

- `test_quality_battery` — indexes `tests/quality/corpus/` (3 VHDL,
  3 docs, 2 Python files; one of the docs is a deliberately generic
  "decoy") and runs the 12-query battery from
  `tests/quality/queries.json`:
  - **top-3 file hits ≥ 10/12** (threshold below the battery size so a
    single borderline query does not fail the build);
  - **every `strict` query hits top-1** (unambiguous, high-confidence
    queries: exact identifiers and clearly distinguishable topics);
  - the per-query report is printed to the CI log for triage.
- `test_quality_embedding_determinism` — the same text embeds to
  **bit-identical** vectors across repeated calls (guards against
  non-deterministic model or ORT upgrades, and that the configured
  model loads at the expected 512 dims).

The corpus is small on purpose: model load + embedding a few dozen
short chunks takes ~1–2 min on a 2-core runner. The heavy full-corpus
battery (e.g. against a real user repository) is a manual exercise,
not a CI gate.

## Why ubuntu-latest only

- ~1.5 GB of model files must be downloadable (Hugging Face); only
  `ubuntu-latest` gets a dedicated job, so the model download happens
  once per CI run regardless of matrix size.
- ONNX CPU inference is the workload under test; the hosted
  ubuntu runner is the reference platform users report against.
- Container/emulated runners (RHEL UBI, arm64-QEMU) are for
  wheel/glibc-floor coverage, not quality; doubling them with a model
  download would slow the whole pipeline.

The job caches the models in `actions/cache`
(key `corvidex-embed-cache-v3` →
`/tmp/corvidex-quality/embed-cache`), so the download
only happens on the first run and after a key bump.

## Running locally

```sh
# one-off (downloads ~1.5 GB of models on first run):
CORVIDEX_RUN_QUALITY=1 uv run pytest tests/test_quality.py -v
```

The models are kept under `<data dir>/embed-cache` — locally
`$TMPDIR/corvidex-quality-data`, or point
`CORVIDEX_QUALITY_DATA=/some/dir` to reuse a pre-populated
`embed-cache/` (same layout fastembed/`huggingface_hub` produce).

## Failure triage

The test prints a per-query table:

```
quality battery (real models, structural chunking):
  sync=9.3s chunks: hdl=9 docs=5 code=3
  [OK ] oddr-semantic        expect=rtl/oddr_output.vhd  top3=[...]
  [ok3] layout-semantic      expect=docs/structure.rst   top3=[...]
  [MISS] ...
  top-3 hits: 11/12 (min 10)
  strict top-1: 7/7
```

- **`[MISS]` on a non-strict query, threshold still met** — borderline;
  look at the `top3` list. Usually fine; consider rewording the query
  or making the corpus file more distinctive.
- **`[MISS]` on a strict query or top-3 below the threshold** — real
  regression. Common causes, in order of likelihood:
  1. embedding model or fastembed version changed (check `uv.lock`);
  2. chunker change split/merged the expected chunk (the battery is
     file-level, so this matters when the distinguishing content
     lands in a *different* chunk of the same file — check
     `top3`/symbols);
  3. hybrid fusion (RRF) weights changed in the vector store;
  4. fixture content drifted (queries are written against the
     fixture text — keep both in the same commit).
- **Determinism test failure** — ORT/model non-determinism: do not
  "fix" by loosening; investigate (new fastembed/ORT version?).

## Maintaining the battery

- **Add a query**: append an entry to `tests/quality/queries.json`
  (`id`, `collection` — `hdl`/`docs`/`code`/`knowledge`, `query`,
  `expect` = repo-relative file, `strict` = bool). Mark `strict` only
  for queries that should *always* hit top-1 (identifiers, unique
  topics). After adding, run the local command above and, if the new
  query is borderline, lower `MIN_TOP3_HITS` or keep it non-strict.
- **Add/change corpus content**: keep topics *distinguishable* and
  identifiers unique per file (the full-text leg leans on identifiers; the
  dense leg on topic separation). Re-run locally.
- **Changing the default model** (the `hdl_model`/`docs_model`/
  `code_model`/`rerank_model` config defaults): re-run locally, re-tune
  thresholds if needed, update the dimension assert in
  `test_quality_embedding_determinism`, and **bump the CI cache key**
  in `.github/workflows/ci.yml`.
- **Reranking/query expansion regressions**: run locally with
  `rerank_enabled=false` / `query_expansion_enabled=false` (see
  `EmbeddingsConfig`) to isolate which stage changed the ranking
  before assuming a model/chunker regression.
- **Tuning thresholds**: only with a local run that shows stable
  results; document the rationale in a commit message.

## Deliberate limitations

- `vhdl_ls`/`veridian` are **not** part of this test (non-existent
  binary paths → deterministic structural chunking fallback), so the
  LSP-symbol chunking path is *not* covered by the quality gate
  (it is covered by the unit tests and the real-binary e2e tests gated
  on `VHDL_LS_TEST_BIN`/`VERIDIAN_TEST_BIN`).
- One corpus, one model layout: the gate catches regressions in the
  default configuration. Non-default `*_model` configs and the
  larger real-world corpus are validated ad hoc (see the model
  comparison notes in the e2e report for this project).
- Quality thresholds are statistical on a small sample; the gate
  trades precision for the ability to run on every push.