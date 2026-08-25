# vhdl-rag-mcp

An MCP (Model Context Protocol) server that gives coding agents
high-quality semantic search over an organization's VHDL code,
VHDL-related documentation, and general source code (C/C++, Python,
...) — all cross-referenced, all with exact source attribution.

Runs as an MCP server over stdio (installed from this Git
repository with `uvx`, see [Installation](#installation)). No
external services required: Qdrant runs embedded and the embedding
models run locally (ONNX via FastEmbed).

## Intended use

The main uses are **RAG** and **cross-referencing code against
documentation**, for coding agents (Claude Code, Maki, or any MCP
client) that implement or modify VHDL.

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
bridge the domains. A standard that says "asynchronous resets are
named `rst_n`" can be checked against the VHDL that actually uses
`rst_n` and the C testbench that drives it; a signal renamed in the
RTL can be found in every doc section and test function that still
references the old name. In practice that means:

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

- **Three indexed domains, one server.** VHDL source, documentation
  (Markdown/reST/text), and general code (C/C++, Python, ...) live in
  three Qdrant collections, each with a dense (jina v2) *and* a sparse
  (BM25) vector per chunk.
- **Hybrid search.** Every query runs Qdrant's native hybrid
  (dense + sparse, RRF-fused) query: semantic similarity *and* exact
  identifier matching in one call. Ask about `rst_n` and you get it.
- **VHDL-aware chunking.** VHDL files are chunked per construct
  (entity, architecture, process, package, function, component) using
  the [vhdl_ls](https://vhdl-lang.org/) language server
  (`documentSymbol` with exact line ranges), with a structural
  line-scanner fallback for files with syntax errors — and a
  whole-file last resort so no VHDL is ever lost.
- **Structure-aware chunking elsewhere.** Documentation is chunked per
  heading section; general code is chunked per top-level
  function/class by tree-sitter (any language with a grammar), with
  file-scope gap chunks for uncovered top-level code.
- **Cross-referencing.** Every chunk payload stores the identifiers it
  defines or references (`symbols`). Search tools accept a `symbols`
  filter that matches chunks referencing the given identifiers —
  bridging docs ↔ VHDL ↔ test code (e.g. find every VHDL process and C
  function that touch `fifo_write`).
- **Exact source attribution.** Every result names repository, file,
  line range, and commit; `get_source` returns the exact current file
  (or a line range) from the synced working tree.
- **Incremental, self-maintaining index.** Repositories are synced
  from Git (clone/fetch/diff): only changed files are re-chunked and
  re-embedded. A background task syncs every `sync_interval` seconds;
  the tools can force a sync or a full reindex at any time.
- **Graceful degradation.** Failures are contained per repository and
  recorded in state; a broken repository never blocks the others or
  the server.
- **Stdout is protocol-clean.** All logging goes to stderr and a
  rotating log file, so the server is safe to run from any MCP host.

## Installation

Requirements:

- [uv](https://docs.astral.sh/uv/) (for `uvx`), Python ≥ 3.12
- Git (with your normal credentials/SSH setup for private repos)
- The `vhdl_ls` binary (only needed for repositories that contain
  VHDL): install a release from
  <https://vhdl-lang.org/> so `vhdl_ls` is on your `PATH`, or point
  `vhdl_ls_path` at the binary. The `vhdl_libraries` directory shipped
  next to the binary is auto-detected. Per repository, `vhdl_ls_hook`
  may generate the `vhdl_ls.toml` workspace config (below); otherwise
  the server writes a built-in default.

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
embedding models (jina v2 `base-code` + `base-en`, ~tens of MB each,
once), and performs an initial sync of all configured repositories.

## Configuration

Config file: `~/.config/vhdl-rag/config.toml` (created with a
commented template on first run if absent).

```toml
data_dir = "~/.local/share/vhdl-rag"   # all state lives here
sync_interval = 300                    # seconds between periodic syncs
vhdl_ls_path = "vhdl_ls"               # binary on PATH or full path
log_level = "INFO"

# [qdrant]
# mode = "local"                       # embedded (default) — or "server" with url
# url = "http://qdrant:6333"

[[repositories]]
name = "company-standards"             # unique, [A-Za-z0-9._-]
url = "git@github.com:company/vhdl-standards.git"
ref = "main"                           # branch (tracked on every sync),
                                       # tag, or commit SHA (pinned)
# domains = ["vhdl", "docs", "code"]   # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]# glob path excludes ('*' crosses '/');
                                       # wildcard-free patterns exclude the subtree
# vhdl_ls_hook = "make vhdl-ls-config" # command run at the repo root to
                                       # generate vhdl_ls.toml when missing

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
  `--sync-interval`, `--vhdl-ls-path`, `--log-level`); the command
  line wins.
- **`url` or `path`** (exactly one): `url` is a remote Git repository,
  cloned and kept in sync by the server under `data_dir/repos`.
  `path` is a **local working repository** — your own checkout, indexed
  in place and never modified (no clone, fetch, or checkout by the
  server).
- **`ref`**: a branch name is fetched and tracked on every sync. A tag
  or commit SHA pins the repository (a full 40-hex SHA skips the
  network fetch entirely). `ref` is ignored for local working
  repositories.
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
  commit. Deleting an untracked file is not tracked between syncs —
  `reindex` repairs it.
- **Per-repository domains/excludes**: index only what a repository
  should contribute — e.g. `domains = ["vhdl"]` for a pure IP
  repository, `exclude = ["sim"]` to skip simulation-only files.
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
background task syncs all repositories every `sync_interval` seconds.
A single-instance lock (`data_dir/server.lock`) prevents two servers
from sharing one data directory.

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
| `search_vhdl(query, limit, repository, symbols)` | Hybrid search over VHDL source (entities, architectures, processes, packages, functions). |
| `search_docs(...)` | Same over documentation sections. |
| `search_code(...)` | Same over general code units (functions/classes). |
| `search_knowledge(query, limit, ...)` | All three domains at once, RRF-fused. |
| `get_source(repository, file, start_line, end_line)` | Exact current file content (or a slice) with commit attribution. |
| `repository_status()` | Per repository: ref, domains, last indexed commit, last sync, last error. |
| `sync_repositories(repositories?)` | Incremental sync (default: all). Failures contained per repository. |
| `reindex_repository(repository)` | Drop and rebuild one repository's index. |

All search tools take an optional `repository` (name) filter plus
`symbols: list[str]` — restrict results to chunks referencing any of
the given identifiers.
Results are rendered as markdown with source attribution, score, and
referenced identifiers; content is fenced by domain.

Example agent flow:

1. `search_knowledge("asynchronous reset conventions")` → a docs
   section plus VHDL processes that implement resets.
2. `search_vhdl("reset", symbols=["rst_n"])` → every VHDL chunk
   touching `rst_n`.
3. `get_source("company-standards", "rtl/reset_ctrl.vhd", 12, 40)` →
   the exact lines to copy.

## Operations

- **Data directory** (`data_dir`): Qdrant collections, the per-repo
  Git working trees (`<name>/`), sync state
  (`state/repositories.json`), the log file (`logs/vhdl-rag-mcp.log`),
  and the lock file. Deleting it resets the index.
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

The test suite runs fully offline: local `file://` git remotes, a fake
LSP server script, and fake embedding providers (one real-binary test
is gated on the `VHDL_LS_TEST_BIN` environment variable).

CI (`.github/workflows/ci.yml`) runs on every push to `main` and on
pull requests: `ruff format --check`, `ruff check`, `mypy` (strict),
and the full test suite on a matrix of Python 3.12/3.13/3.14 across
Ubuntu, Windows, and macOS, plus an AlmaLinux 9 container job.

Layout:

```
src/vhdl_rag_mcp/
  config.py        typed config (pydantic) + default template
  state.py         atomic repository sync state
  git_manager.py   async clone/fetch/checkout + incremental SyncPlan
  routing.py       extension -> domain classification (+domains/excludes)
  lsp/client.py    vhdl_ls LSP client (handshake, quiet-wait, symbols)
  embeddings/      FastEmbed dense/sparse providers (per-collection + shared)
  vector_store.py  Qdrant wrapper: hybrid RRF query, payload filters
  indexing/        vhdl (LSP-primary), docs (sections), code (tree-sitter),
                   pipeline (incremental sync driver)
  retrieval.py     search service: fusion, source access
  server.py        FastMCP tools + startup + periodic sync + lock
```