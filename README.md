# vhdl-rag-mcp

An MCP (Model Context Protocol) server that gives coding agents
high-quality semantic search over an organization's HDL code (VHDL,
Verilog, SystemVerilog), HDL-related documentation, and general source
code (C/C++, Python, ...) — all cross-referenced, all with exact
source attribution.

Runs as an MCP server over stdio (installed from this Git
repository with `uvx`, see [Installation](#installation)). No
external services required: the vector store (SQLite + the
sqlite-vec extension) runs embedded and the embedding models run
locally (ONNX via FastEmbed).

## Intended use

The main uses are **RAG** and **cross-referencing code against
documentation**, for coding agents (Claude Code, Maki, or any MCP
client) that implement or modify HDL (VHDL, Verilog, or
SystemVerilog).

**RAG (Retrieval-Augmented Generation).** RAG is a technique for
keeping a language model grounded in *your* material instead of only
its training data: before (or while) the model generates an answer, it
first *retrieves* relevant chunks from a knowledge base and uses those
as context. For a coding agent, that means the context it needs
usually lives *outside* the file it is editing — the company's coding
standards, design guides, and reference IP from earlier projects.
`vhdl-rag-mcp` is that retrieval layer: it maintains an up-to-date,
semantically searchable index of your repositories and hands the
agent the verbatim text (with exact repository, file, line range, and
commit) of every match, so the agent grounds its work in what the
organization actually wrote instead of hallucinating a plausible
pattern.

**Cross-referencing code against documentation.** This is what makes
the search more than three separate indexes: every chunk stores the
identifiers it defines or references (`symbols`), so the agent can
bridge the domains — and the HDL languages: a constant shared by a
SystemVerilog package, a Verilog module, and a VHDL entity is found
once and resolves to all of them. A standard that says
"asynchronous resets are named `rst_n`" can be checked against the
VHDL that actually uses `rst_n` and the C testbench that drives it; a
signal renamed in the RTL can be found in every doc section and test
function that still references the old name. In practice that means:

- **Docs → code.** Follow a convention from the standard to every
  VHDL construct and test function that implements it.
- **Code → docs.** Find the design rationale behind an implementation:
  given a process or function, which documentation section explains
  its convention.
- **Consistency.** Trace one identifier (e.g. `wr_ptr`) across
  standard, RTL, and testbench so a rename or protocol change doesn't
  leave the domains out of sync.

Both uses rely on the index staying current: repositories are Git
synced (branch-tracked or pinned to a tag/SHA) automatically in the
background, so the context an agent retrieves reflects the code as it
is, not a stale snapshot.

## Capabilities

- **Three indexed domains, one server.** HDL source (VHDL, Verilog,
  and SystemVerilog in one `hdl` collection, each chunk tagged with
  its language), documentation (Markdown/reST/text), and general code
  (C/C++, Python, ...) live in three SQLite collections (a vec0
  vector table + an FTS5 full-text table each). Every query runs a
  hybrid (dense + full-text, RRF-fused) search: semantic similarity
  *and* exact identifier matching in one call. Ask about `rst_n` and
  you get it.
- **HDL-aware chunking.** VHDL files are chunked per construct
  (entity, architecture, process, package, function, component) using
  the [vhdl_ls](https://vhdl-lang.org/) language server
  (`documentSymbol` with exact line ranges); Verilog and SystemVerilog
  are chunked by [Veridian](https://github.com/chipsalliance/veridian)
  (module/program/interface, package, inner functions and tasks,
  normalized to the same cross-language model — module → `design_unit`,
  `always_ff` → `process` — with the server-native kind kept as
  `native_symbol_kind`). Both have a structural line-scanner fallback
  for files with syntax errors, and a whole-file last resort so no HDL
  is ever lost.
- **Structure-aware chunking elsewhere.** Documentation is chunked per
  heading section; general code is chunked per top-level
  function/class by tree-sitter (any language with a grammar), with
  file-scope gap chunks for uncovered top-level code.
- **Cross-referencing.** Every chunk payload stores the identifiers it
  defines or references (`symbols`). Search tools accept a `symbols`
  filter that matches chunks referencing the given identifiers —
  bridging docs ↔ HDL ↔ test code (e.g. find every construct that
  touches `fifo_write`), and across HDL languages (e.g. `FIFO_DEPTH`
  in a VHDL generic, a Verilog localparam, and an SV package constant).
- **Optional HDL analyzers, graceful degradation.** `vhdl_ls` and
  Veridian are external binaries that are *not* bundled or installed by
  this server: each is located via its config path or on `PATH`, and
  when one is missing its files simply fall back to structural/generic
  parsing. `repository_status` reports each analyzer's availability,
  version, and mode (`lsp` or `fallback`).
- **Exact source attribution.** Every result names repository, file,
  line range, and commit; `get_source` returns the exact current file
  (or a line range) from the synced working tree.
- **Incremental, self-maintaining index.** Repositories are synced
  from Git (clone/fetch/diff): only changed files are re-chunked and
  re-embedded. A background task syncs every `sync_interval` seconds;
  the tools can force a sync or a full reindex at any time. Local
  working repositories get a fast change poller (`local_sync_interval`,
  default 10 s) so in-progress work — commits, tracked edits, untracked
  file add/remove — lands in the index within about one poll, read-only
  and never interfering with the user's checkout.
- **Graceful degradation.** Failures are contained per repository and
  recorded in state; a broken repository never blocks the others or
  the server. A missing language-server binary degrades that analyzer
  to structural parsing (see above) instead of failing.
- **Stdout is protocol-clean.** All logging goes to stderr and a
  rotating log file, so the server is safe to run from any MCP host.

## Installation

Requirements:

- [uv](https://docs.astral.sh/uv/) (for `uvx`), Python ≥ 3.12
- Git (with your normal credentials/SSH setup for private repos)
- Supported platforms: Linux with glibc ≥ 2.34 (RHEL 9/10 and
  derivatives such as AlmaLinux 9.6+, Ubuntu 24.04, Debian 12;
  x86_64 and arm64), Windows, and macOS 14+ (Apple Silicon and
  Intel). CI verifies all three OS families, including arm64 Linux,
  on Python 3.12–3.14.
- `vhdl_ls` (only needed for repositories that contain VHDL): install
  a release from <https://vhdl-lang.org/> so `vhdl_ls` is on your
  `PATH`, or point `vhdl_ls_path` at the binary. The
  `vhdl_libraries` directory shipped next to the binary is
  auto-detected. Per repository, `vhdl_ls_hook` may generate the
  `vhdl_ls.toml` workspace config (below); otherwise the server writes
  a built-in default.
- Veridian (only needed for repositories that contain Verilog or
  SystemVerilog): install it so `veridian` is on your `PATH`, or point
  `veridian_path` at the binary. Per repository, `veridian_hook` may
  generate the `veridian.yaml` workspace config (below); otherwise the
  server writes a built-in default that declares the repository root as
  the workdir and include/source roots, so `` `include ``/`` `define ``
  resolve in-tree.
- Both binaries are **optional**: without one, its files are indexed
  with a structural/generic fallback instead.

The package is installed from this Git repository (it is not on
PyPI):

```console
$ uvx --from git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git vhdl-rag-mcp
```

`uvx` supports branch/tag pins in the same syntax:
`git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git@v1.0`. The server
accepts `--help`:

```console
$ uvx --from git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git vhdl-rag-mcp --help
```

On first start the server creates its data directory, downloads the
embedding model (jina v2 `small-en`, ~0.1 GB, once), and performs an
initial sync of all configured repositories.

## Configuration

Config file: `~/.config/vhdl-rag/config.toml` (created with a
commented template on first run if absent).

```toml
data_dir = "~/.local/share/vhdl-rag"   # all state lives here
sync_interval = 300                    # seconds between periodic syncs
local_sync_interval = 10               # fast poller for local working
                                       # repositories (0 disables; remote
                                       # repositories ignore it)
vhdl_ls_path = "vhdl_ls"               # binary on PATH or full path (VHDL)
veridian_path = "veridian"             # binary on PATH or full path (Verilog/SV)
log_level = "INFO"

# [embeddings]
# dense_max_tokens = 1024              # passages are truncated to this many
#                                      # tokens before dense embedding (model
#                                      # context is 8192; attention memory is
#                                      # quadratic in length)
# dense_threads = 4                    # CPU threads for dense ONNX inference
# dense_enable_cpu_mem_arena = false   # ONNX CPU memory arena (fast but
#                                      # retains peak buffers, ~+2.5 GB;
#                                      # false = lower RAM, ~35% slower)
# dense_batch_size = 1                 # passages per ONNX inference call
#                                      # (1 = strict per-passage memory
#                                      # bound; higher trades memory for speed)
# index_max_tokens = 512               # indexed passages are truncated to
#                                      # this many tokens (queries are
#                                      # unaffected; must be <= dense_max_tokens)
# indexing_workers = 1                 # worker processes for data-parallel
#                                      # dense embedding (1 = single process;
#                                      # each worker loads its own model copy)
# hdl_model = "jinaai/jina-embeddings-v2-small-en"  # any fastembed
# docs_model = "jinaai/jina-embeddings-v2-small-en"  # TextEmbedding name;
# code_model = "jinaai/jina-embeddings-v2-small-en"  # default: small-en (512 dims)

[[repositories]]
name = "company-standards"             # unique, [A-Za-z0-9._-]
url = "git@github.com:company/vhdl-standards.git"
ref = "main"                           # branch (tracked on every sync),
                                       # tag, or commit SHA (pinned)
# domains = ["hdl", "docs", "code"]     # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]  # glob path excludes ('*' crosses '/');
                                        # wildcard-free patterns exclude the subtree
# vhdl_ls_hook = "make vhdl-ls-config"  # command run at the repo root to
                                        # generate vhdl_ls.toml when missing
# veridian_hook = "make veridian-config"  # command to generate veridian.yaml

# ... or index your own active checkout instead of a remote:
[[repositories]]
name = "current-project"
path = "~/work/current-project"        # local working repository
```

Notes:

- **Config file selection**: the default location is
  `~/.config/vhdl-rag/config.toml` (a commented template is written
  there on first run). Select another file with the `VHDL_RAG_MCP_CONFIG`
  environment variable or the `--config PATH` flag. The top-level scalar
  options also have command-line overrides (`--data-dir`,
  `--sync-interval`, `--local-sync-interval`, `--vhdl-ls-path`,
  `--veridian-path`, `--log-level`); the command line wins.
- **`url` or `path`** (exactly one): `url` is a remote Git repository,
  cloned and kept in sync by the server under `data_dir/repos`.
  `path` is a **local working repository** — your own checkout, indexed
  in place and never modified (no clone, fetch, or checkout by the
  server). Local repositories are additionally watched by a fast
  poller: every `local_sync_interval` seconds (default 10, 0 disables
  it) the server computes a read-only fingerprint of the working tree
  (HEAD + `git status` porcelain) and syncs the repository when it
  changed — so commits, tracked edits, and untracked file add/remove
  show up in the index within about one poll. Untracked file content
  is fingerprinted at sync time: unchanged files are not re-chunked,
  edited files are re-chunked, and deleted untracked files are dropped
  from the index.
- **`ref`**: a branch name is fetched and tracked on every sync. A tag
  or commit SHA pins the repository (a full 40-hex SHA skips the
  network fetch entirely). `ref` is ignored for local working
  repositories.
- **`[embeddings]`**: dense-inference bounds (memory safety and speed).
  `dense_max_tokens` (default 1024, maximum 8192) truncates a passage
  before dense embedding; `dense_threads` (default 4) caps the ONNX
Runtime thread pool. ONNX Runtime arenas retain peak tensor sizes and
    attention work is quadratic in sequence length, so without these
    bounds a single long chunk can make one embedding batch reserve tens
    of GB. `dense_enable_cpu_mem_arena` (default false) additionally
    disables the ONNX CPU memory arena: buffers are released after each
    inference, halving peak RAM (measured 5.4 → 2.9 GB) at a ~35%
    indexing-time cost — set true when indexing speed matters more and
    RAM is plentiful. `dense_batch_size` (default 1) limits passages
    per ONNX inference call — at 1, peak inference memory is bounded by
    a single truncated passage regardless of batch content (raise it
    only when throughput matters more than memory). Indexing itself
    embeds and upserts in bounded streams (256 chunks per round), so
    resident passage/vector buffers never grow with repository size.
    `index_max_tokens` (default 512) truncates
    *indexed* passages before embedding (queries are unaffected — they
    are short): on the measured corpus quality is unchanged at 512 while
    indexing is faster and lighter. `indexing_workers` (default 1) runs
    data-parallel embedding with N worker processes during indexing
    (each loads its own model copy, ~0.1 GB for the default); quality
    is identical. `hdl_model`/`docs_model`/`code_model` (default: jina
    v2 small-en, 512 dims — measured equal-or-better retrieval quality
    at ~3x lower indexing RAM) select the dense model per collection;
    any fastembed `TextEmbedding` model name works. Switching to a
    model with a different vector size requires a reindex (delete the
    collection or `data_dir`). Computed dense vectors are cached
    content-addressed under `data_dir/dense-cache`, so reindexing, branch
    flips, and duplicated content skip re-embedding.
- **`vhdl_ls_hook`**: shell command run at the repository root that
  generates `vhdl_ls.toml` when the file is missing (before the
  `vhdl_ls` session for that repository). When no hook is set, the hook
  fails, or it leaves no file behind, the server writes a built-in
  default (a `defaultlib` glob for all `.vhd`/`.vhdl` files plus the
  standard libraries shipped with `vhdl_ls`) and removes it after the
  session; files a hook creates are owned by the hook and are never
  removed by the server. For local working repositories the hook runs
  inside your own checkout.
- **Local working repositories** index the working tree: HEAD plus
  uncommitted changes (staged and unstaged) and untracked files
  (honoring `.gitignore`); chunks are attributed to the current HEAD
  commit.
- **Per-repository domains/excludes**: index only what a repository
  should contribute — e.g. `domains = ["hdl"]` for a pure IP
  repository (`"vhdl"` is accepted as a legacy alias for `"hdl"`),
  `exclude = ["sim"]` to skip simulation-only files.
- **Changing embedding models** changes the dense vector dimension;
  the server fails loudly with an actionable message instead of
  corrupting the index (delete the collection or `data_dir` and
  reindex).

## Usage

### Run the server

```console
$ uvx --from git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git vhdl-rag-mcp
```

It serves MCP over stdio until the host closes the connection; a
background task syncs all repositories every `sync_interval` seconds,
and local working repositories are additionally change-checked every
`local_sync_interval` seconds by a fast, read-only poller.
A single-instance lock (`data_dir/server.lock`) prevents two servers
from sharing one data directory.

At startup the server runs a self-check of its runtime components —
`git`, the SQLite runtime (including FTS5), the `sqlite-vec`
extension, the index schema version, the per-collection embedding
models, and the HDL analyzers — and logs a one-line summary
(`startup self-check: ok` or a list of what is degraded). Missing
*required* components (git, FTS5, sqlite-vec) abort startup with an
actionable error; missing *optional* components degrade gracefully:
without `vhdl_ls`/Veridian the affected files fall back to structural
parsing, and without an embedding model that collection's embedding
search and indexing are unavailable (lexical search still works) until
the model is provisioned. `repository_status` reports the current
component state (analyzers + embedding models).

### Register with an MCP client

Claude Code:

```console
$ claude mcp add vhdl-rag-mcp -- uvx --from git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git vhdl-rag-mcp
```

Maki (TOML config — verify the exact table names against your Maki
version's docs):

```toml
[mcp_servers.vhdl_rag_mcp]
command = "uvx"
args = ["--from", "git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git", "vhdl-rag-mcp"]
```

### Tools

| Tool | What it does |
| --- | --- |
| `search_hdl(query, limit, repository, symbols, language, mode)` | Search over HDL source (VHDL, Verilog, SystemVerilog): design units (entities/modules), architectures, processes/always blocks, packages, functions, tasks. `language` filters by HDL language. |
| `search_vhdl(query, limit, repository, symbols)` | `search_hdl` restricted to VHDL (back-compat name). |
| `search_docs(...)` | Same over documentation sections. |
| `search_code(...)` | Same over general code units (functions/classes). |
| `search_knowledge(query, limit, ...)` | All three domains at once, RRF-fused. |
| `get_source(repository, file, start_line, end_line)` | Exact current file content (or a slice) with commit attribution. |
| `repository_status()` | Per repository: ref, priority, domains, last indexed commit, last sync, last error — plus the HDL analyzer status (`vhdl_ls`, Veridian: available, version, `lsp`/`fallback` mode) and the per-collection embedding-model state. |
| `sync_repositories(repositories?)` | Incremental sync (default: all). Failures contained per repository. |
| `reindex_repository(repository)` | Drop and rebuild one repository's index. |

All search tools take an optional `repository` (name) filter plus
`symbols: list[str]` — restrict results to chunks referencing any of
the given identifiers. `search_hdl`/`search_knowledge` additionally
accept `language` (e.g. `"verilog"`) to restrict results by language.
Every search tool also takes `mode`: `hybrid` (default; semantic +
full-text, RRF-fused), `semantic` (embedding similarity only), or
`lexical` (full-text match only; no embedding involved). Each
repository's `priority` (config, default 1) applies a bounded
post-RRF bonus so higher-priority repositories rank slightly ahead of
equally relevant chunks elsewhere without ever crossing relevance
tiers.
Results are rendered as markdown with source attribution, score,
language, and referenced identifiers; HDL content is fenced by
language.

Example agent flow:

1. `search_knowledge("asynchronous reset conventions")` → a docs
   section plus VHDL and Verilog constructs that implement resets.
2. `search_hdl("reset", symbols=["rst_n"])` → every HDL chunk touching
   `rst_n`, in every HDL language.
3. `search_hdl("fifo", language="systemverilog")` → only SystemVerilog.
4. `get_source("company-standards", "rtl/reset_ctrl.vhd", 12, 40)` →
   the exact lines to copy.

## Operations

- **Data directory** (`data_dir`): the SQLite index (`index.sqlite`), the per-repo
  Git working trees (`<name>/`), sync state
  (`state/repositories.json`), the log file (`logs/vhdl-rag-mcp.log`),
  the model cache (`embed-cache`), the dense-vector cache
  (`dense-cache`), and the lock file. Deleting it resets the index.
- **State & retries**: a repository's `indexed_commit` advances only
  after its index update fully succeeded; a failed sync keeps the
  previous commit and the next sync retries the same diff.
  `last_sync_error` is visible via `repository_status`.
- **Removing a repository** from the config: on the next start the
  server detects it in the state file and automatically drops all of
  its chunks and state.
- **Logs**: `stderr` + `logs/vhdl-rag-mcp.log` (rotating, 3×5 MB).
  `log_level = "DEBUG"` for LSP/git/embedding detail.

## Development

```console
$ uv sync
$ uv run ruff format -q . && uv run ruff check .   # format + lint
$ uv run mypy src                                   # strict types
$ uv run pytest -q                                  # offline test suite
```

The test suite runs fully offline: local `file://` git remotes, fake
LSP server scripts (vhdl_ls *and* Veridian), and fake embedding
providers (real-binary tests are gated on the `VHDL_LS_TEST_BIN` and
`VERIDIAN_TEST_BIN` environment variables).

CI (`.github/workflows/ci.yml`) runs on every push to `main` and on
pull requests: `ruff format --check`, `ruff check`, `mypy` (strict),
and the full test suite on a matrix of Python 3.12/3.13/3.14 across
Ubuntu, Windows, and macOS, plus RHEL 9 and RHEL 10 container jobs
(official UBI images; UBI 9's glibc 2.34 is the strictest floor in
the dependency wheel set), and linux/arm64 jobs that run the suite
in manylinux aarch64 containers (glibc 2.34 and 2.39 floors,
mirroring the RHEL jobs) under QEMU user-mode emulation (no hosted
arm64 runners; this verifies the aarch64 wheels and execution on the
architecture).

Layout:

```
src/vhdl_rag_mcp/
  config.py        typed config (pydantic) + default template
  state.py         atomic repository sync state (schema-versioned)
  git_manager.py   async clone/fetch/checkout + incremental SyncPlan
  routing.py       extension -> domain classification (+domains/excludes)
  lsp/             LSP transport (server-agnostic) + vhdl_ls and Veridian
                   adapters + analyzer discovery/status
  embeddings/      FastEmbed dense providers (per-collection, lazy)
  vector_store.py  sqlite-vec wrapper: hybrid (dense + FTS5) RRF query,
                   row filters
  indexing/        vhdl (vhdl_ls), verilog (Veridian), docs (sections),
                   code (tree-sitter), pipeline (incremental sync driver)
  retrieval.py     search service: fusion, language filter, source access
  server.py        FastMCP tools + startup + periodic sync + lock
```