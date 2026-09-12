# Configuration reference

The [README](../README.md#quick-start) covers the zero-config default:
with no `[[repositories]]` configured at all, the directory the server
is started in is indexed automatically (`index_cwd`, default `true`).
This document covers everything beyond that default — indexing more
than one repository, a coding-standards file, tuning the embedding
pipeline, air-gapped installs, and the full set of notes on repository
semantics (local checkouts, submodules, hooks, etc.).

## Config file

Location: a project-local `.corvidex` file in the project directory —
there is no global config location; every project configures itself,
right next to the code it describes. The file is optional and is never
created implicitly: with no `.corvidex` the built-in defaults apply and
the project directory is indexed. Run `corvidex-mcp --init-config` to
write a commented template when you want one. Select another file with
the `CORVIDEX_MCP_CONFIG` environment variable or the `--config PATH`
flag; either always wins over `.corvidex`.
The top-level scalar options also have command-line overrides
(`--data-dir`, `--sync-interval`, `--local-sync-interval`,
`--vhdl-ls-path`, `--vhdl-ls-libraries-dir`, `--veridian-path`,
`--log-level`, `--no-index-cwd`, `--num-threads`); the command line
wins. `--vhdl-ls-libraries-dir` in particular describes the *vhdl_ls
install* rather than any one project (a `cargo install`ed vhdl_ls ships
without its standard libraries and panics without them), so it belongs
on the launcher command line, where it applies to every workspace.

### Project directory

The project directory — where `.corvidex` is looked for, and what gets
indexed with no configuration — is the **working directory the server
was started in**, since a stdio MCP server inherits its cwd from the
agent that spawned it.

A launcher can break that inheritance and make corvidex index *itself*
instead of your code. The usual culprit is `uv --directory DIR run`,
which changes the working directory to the corvidex checkout; use
`uv --project DIR run` instead, which selects the same environment but
leaves the cwd alone:

```jsonc
// wrong: indexes corvidex-mcp's own source tree
"args": ["--directory", "/path/to/corvidex-mcp", "run", "corvidex-mcp"]
// right: indexes the agent's workspace
"args": ["--project", "/path/to/corvidex-mcp", "run", "corvidex-mcp"]
```

When a launcher cannot be fixed, set `CORVIDEX_MCP_PROJECT_DIR` to the
workspace root and it wins over the working directory. The server logs a
warning if it ever auto-indexes its own source tree.

## Full example

```toml
# data_dir = "~/.local/share/corvidex"  # one shared store for every project;
                                        # always used once [[repositories]] is
                                        # set below. With no [[repositories]]
                                        # (zero-config), leave this unset and
                                        # each auto-indexed project instead
                                        # gets its own storage subdirectory
                                        # under project_data_root, so unrelated
                                        # projects never share an index.sqlite
                                        # or sync state; set data_dir here to
                                        # opt back into one shared store.
# project_data_root = "~/.local/share/corvidex/projects"  # per-project
                                        # storage root (zero-config only,
                                        # ignored if data_dir is set or
                                        # [[repositories]] is configured)
sync_interval = 300                    # seconds between periodic syncs
local_sync_interval = 10               # fast poller for local working
                                       # repositories (0 disables; remote
                                       # repositories ignore it)
vhdl_ls_path = "vhdl_ls"               # binary on PATH or full path (VHDL)
veridian_path = "veridian"             # binary on PATH or full path (Verilog/SV)
log_level = "INFO"
# index_cwd = false                    # disable the zero-config default
                                       # (index the server's cwd when no
                                       # [[repositories]] are configured);
                                       # with this set and no repositories,
                                       # the server runs with an empty index
# coding_standards = "~/standards/standards.md"  # the coding-standards file
#                                                # (txt, md, rst, pdf, docx);
#                                                # indexed as the
#                                                # 'coding-standards'
#                                                # pseudo-repository
# coding_standards_priority = 10           # retrieval priority of that file
                                           # (bounded post-RRF bonus; 10 ranks
                                           # it ahead of equally relevant repo
                                           # chunks, relevance still dominates)

# [embeddings]
# dense_max_tokens = 1024              # passages are truncated to this many
#                                      # tokens before dense embedding (model
#                                      # context is 8192; attention memory is
#                                      # quadratic in length)
# dense_threads = 4                    # CPU threads for dense ONNX inference
#                                      # (default: half the host's CPU count,
#                                      # at least 1; override with
#                                      # --num-threads)
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
# hdl_model = "jinaai/jina-embeddings-v2-small-en"  # any fastembed TextEmbedding
# docs_model = "jinaai/jina-embeddings-v2-small-en"  # name; hdl/docs default to
#                                                     # small-en (512 dims); code
# code_model = "jinaai/jina-embeddings-v2-base-code"  # defaults to base-code
#                                                      # (768 dims, code-pretrained)

# With any [[repositories]] configured, the zero-config cwd default no
# longer applies — these are indexed instead.
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

  # ... or index another active checkout instead of a remote:
[[repositories]]
name = "another-project"
path = "~/work/another-project"        # local working repository
# index_untracked = false             # skip untracked files (default: index them)

  # ... or index a plain directory of files with no Git at all:
[[repositories]]
name = "local-ip"
path = "~/work/local-ip"               # plain directory, no Git
filesystem = true
```

## Notes

- **Zero-config default (`index_cwd`)**: when the configuration has no
  `[[repositories]]` at all (the common case: no config file, or one
  that only sets scalar options), the directory the server is started
  in is indexed automatically — a Git working tree as a local working
  repository (HEAD plus uncommitted and untracked changes), a plain
  directory as a filesystem repository. Its name is derived from the
  directory name (sanitized to `[A-Za-z0-9._-]`, falling back to
  `workspace`) plus a short hash of the resolved path (e.g.
  `backend-a1b2c3d4`), so two different directories that happen to
  share a basename never collide; use `repository_status` to see the
  exact name assigned. Set `index_cwd = false` (or pass
  `--no-index-cwd`) to disable this and run with an empty index instead.
  Adding even one `[[repositories]]` entry also disables it — the
  default only applies when repositories is empty.
- **Per-project storage (zero-config only)**: because zero-config is
  meant to "just work" when the server is started in different
  projects over time, each auto-indexed project gets its own
  `index.sqlite` and sync state in a subdirectory of `project_data_root`
  (default `~/.local/share/corvidex/projects`), named after the
  auto-derived repository (see above, already collision-proof). Models
  and the dense-vector cache stay shared under `data_dir` regardless
  (they are large, content-addressed, and safe to reuse across
  projects). Setting `data_dir` explicitly opts back into a single
  shared index across every project — and once any `[[repositories]]`
  entry is configured, the shared store is always used (so
  `search_knowledge` can search across all configured repositories from
  one index).
- **Config file selection**: the default (and only on-disk) location is
  a project-local `.corvidex` file in the project directory, which is
  optional and never written implicitly (`--init-config` writes a
  commented template on request) — there is no global config file.
  Select another file with the
  `CORVIDEX_MCP_CONFIG` environment variable or the `--config PATH` flag
  — either wins over `.corvidex`. The top-level scalar options also have
  command-line overrides (`--data-dir`, `--sync-interval`,
  `--local-sync-interval`, `--vhdl-ls-path`,
  `--vhdl-ls-libraries-dir`, `--veridian-path`, `--log-level`,
  `--no-index-cwd`); the command line wins.
- **`url` or `path`** (exactly one): `url` is a remote Git repository,
  cloned and kept in sync by the server under `data_dir/repos`.
  `path` is a **local working repository** — your own checkout, indexed
  in place and never modified (no clone, fetch, or checkout by the
  server). `path` with `filesystem = true` is a **filesystem
  repository** — a plain directory of files with no Git involved at
  all: every file below `path` is walked and indexed in place (hidden
  files/directories and symlinks are skipped, so an embedded `.git`
  directory never enters the index). Incremental sync re-walks the
  directory (paths + mtimes + sizes) and fingerprints file content, so
  edits, additions, and deletions are picked up on the next sync; the
  fast local poller watches these repositories as well. `ref` and the
  Git hooks do not apply. Local repositories are additionally watched by a fast
  poller: every `local_sync_interval` seconds (default 10, 0 disables
  it) the server computes a read-only fingerprint of the working tree
  (HEAD + `git status` porcelain) and syncs the repository when it
  changed — so commits, tracked edits, and untracked file add/remove
  show up in the index within about one poll. Untracked file content
  is fingerprinted at sync time: unchanged files are not re-chunked,
  edited files are re-chunked, and deleted untracked files are dropped
  from the index.
- **`coding_standards`**: one file with the organization's coding
  standards (txt, md, rst, pdf, or docx). It is indexed as the
  `coding-standards` pseudo-repository in the docs collection with a
  high retrieval priority (`coding_standards_priority`, default 10):
  standards chunks rank ahead of equally relevant repository chunks
  (bounded post-RRF bonus — relevance still dominates). The file's
  content hash is its "commit", so edits are picked up on the next
  sync. Search it with `repository="coding-standards"`.
- **`ref`**: a branch name is fetched and tracked on every sync. A tag
  or commit SHA pins the repository (a full 40-hex SHA skips the
  network fetch entirely). `ref` is ignored for local working
  repositories.
- **`[embeddings]`**: dense-inference bounds (memory safety and speed).
  `dense_max_tokens` (default 1024, maximum 8192) truncates a passage
  before dense embedding; `dense_threads` (default: half the host's CPU
  count, at least 1; falls back to 4 if undetectable — override with the
  `--num-threads` CLI flag or this setting) caps the ONNX Runtime thread
  pool. ONNX Runtime arenas retain peak tensor sizes and
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
  is identical. `hdl_model`/`docs_model`/`code_model` select the dense
  model per collection; `hdl_model`/`docs_model` default to jina v2
  small-en (512 dims — measured equal-or-better retrieval quality at
  ~3x lower indexing RAM than the base models), while `code_model`
  defaults to jina v2 base-code (768 dims), pretrained on source code
  rather than prose/RTL text. Any fastembed `TextEmbedding` model name
  works. Switching to a
  model with a different vector size requires a reindex (delete the
  collection or `data_dir`). Computed dense vectors are cached
  content-addressed under `data_dir/dense-cache`, so reindexing, branch
  flips, and duplicated content skip re-embedding.
- **`vhdl_ls_hook`**: shell command run at the repository root that
  generates `vhdl_ls.toml` when the file is missing (before the
  `vhdl_ls` session for that repository). When no hook is set, the hook
  fails, or it leaves no file behind, the server writes a built-in
  default and removes it after the session; files a hook creates are
  owned by the hook and are never removed by the server. For local
  working repositories the hook runs inside your own checkout.

  The built-in default is **library-aware**: each indexed file is
  mapped to a VHDL library from the tsfpga/hdl-modules directory
  layout — `modules/<library>/…` (`src`, `rtl`, `test`, `sim`,
  `regs_src`, … all belong to the same library), matched at any depth
  so a vendored submodule's own `modules/` wins — and gets a
  `[libraries.<name>]` section, plus the standard libraries shipped
  with `vhdl_ls`. Files whose layout is not recognised fall back to a
  single `defaultlib`. This is what makes a library-qualified name
  (`cnn_accel.cnn_accel_bias_requant`, the standard instantiation
  style in these projects) resolvable: `vhdl_ls` resolves
  `<library>.<name>` only against a library the configuration
  declares, so a flat single-library config silently returns nothing
  from `find_definition`, `find_references` and `hover_info` for every
  such name. Repositories laid out differently should supply their own
  `vhdl_ls.toml` or a `vhdl_ls_hook`.
- **Local working repositories** index the working tree: HEAD plus
   uncommitted changes (staged and unstaged) and untracked files
   (honoring `.gitignore`); chunks are attributed to the current HEAD
   commit. Untracked indexing can be switched off per repository with
   `index_untracked = false`; untracked files that were indexed before
   the flag was turned off are dropped from the index on the next sync.
 - **Submodules are indexed recursively** (nested up to three levels
   deep). A submodule's files enter the index under their gitlink path
   as prefix, e.g. `ip/rtl/a.vhd`, so they search alongside the top
   repository's own files. For remote repositories the server keeps the
   submodules checked out under `data_dir/repos` (best-effort
   `git submodule update --init --recursive`; a submodule that cannot be
   fetched is skipped with a warning, the rest still syncs): a new
   submodule is indexed wholesale, a pointer move re-chunks the whole
   submodule and drops the files gone at the new SHA, and a removed
   submodule purges its entire prefix. For local working repositories
   the submodule's working tree is indexed in place: a moved pointer or
   a moved submodule HEAD re-chunks the submodule wholesale, otherwise
   only its own tracked changes and untracked files are diffed inside
   it (same `index_untracked` flag applies); a deinitialized or removed
   submodule purges its prefix.
- **Per-repository domains/excludes**: index only what a repository
  should contribute — e.g. `domains = ["hdl"]` for a pure IP
  repository (`"vhdl"` is accepted as a legacy alias for `"hdl"`),
  `exclude = ["sim"]` to skip simulation-only files.
- **Changing embedding models** changes the dense vector dimension;
  the server fails loudly with an actionable message instead of
  corrupting the index (delete the collection or `data_dir` and
  reindex).

## Requirements

- [uv](https://docs.astral.sh/uv/) (for `uvx`), Python ≥ 3.12
- Git (with your normal credentials/SSH setup for private repos)
- Supported platforms: Linux with glibc ≥ 2.34 (RHEL 9/10 and
  derivatives such as AlmaLinux 9.6+, Ubuntu 24.04, Debian 12;
  x86_64 and arm64), Windows, and macOS 14+ (Apple Silicon and
  Intel). CI verifies Ubuntu and Windows on Python 3.12–3.14,
  RHEL 9/10 on all three, and macOS on CPython 3.14; arm64 Linux
  is not CI-verified (no hosted runners). The vector store also
  requires an interpreter whose stdlib SQLite supports loadable
  extensions (sqlite-vec): some uv standalone builds (e.g. macOS
  3.12/3.13) lack it — the startup self-check reports this, and
  the fix is `uv python install 3.14`.
- `vhdl_ls` (only needed for repositories that contain VHDL): install
  a release from <https://vhdl-lang.org/> so `vhdl_ls` is on your
  `PATH`, or point `vhdl_ls_path` at the binary. The
  `vhdl_libraries` directory shipped next to the binary is
  auto-detected. Per repository, `vhdl_ls_hook` may generate the
  `vhdl_ls.toml` workspace config (above); otherwise the server writes
  a built-in, library-aware default.
- Veridian (only needed for repositories that contain Verilog or
  SystemVerilog): install it so `veridian` is on your `PATH`, or point
  `veridian_path` at the binary. Per repository, `veridian_hook` may
  generate the `veridian.yaml` workspace config (above); otherwise the
  server writes a built-in default that declares the repository root as
  the workdir and include/source roots, so `` `include ``/`` `define ``
  resolve in-tree.
- Both binaries are **optional**: without one, its files are indexed
  with a structural/generic fallback instead.

On first start the server creates its data directory, loads the
embedding model (jina v2 `small-en`, ~0.1 GB), and starts an initial
sync of all configured (or auto-detected) repositories in the
background. By default the model is downloaded once and cached; it
can alternatively ship inside the installed package so no runtime
download is needed (air-gapped installs — see below).

## Installation

### From PyPI (models downloaded on first run)

```console
$ pip install corvidex-mcp
# or, to run it without installing anything permanently:
$ uvx corvidex-mcp
```

The published wheel is **slim** (1.3 MB): it carries the tokenizer and
config files of every model but none of their ONNX weights. On first
start fastembed downloads the weights the configuration asks for into
`<data_dir>/embed-cache` (~0.86 GB for the three defaults) and every
later start reads them from there.

`pip install corvidex-mcp` needs network access at install time *and*
at first run. For a host that has neither, use the offline bundle
below.

Platform note: on Apple Silicon, **Python 3.14 requires macOS 14+** —
every `onnxruntime` wheel for CPython 3.14 is tagged
`macosx_14_0_arm64`, and unlike 3.12/3.13 there is no older cp314 build
to fall back to. On 3.12/3.13 the resolver backtracks to an
onnxruntime with a macOS 13-compatible wheel.

### Air-gapped installation (offline bundle)

The offline bundle carries **every** model inside the wheel, so an
air-gapped host downloads nothing at install time or at run time. It is
one archive:

```
corvidex-mcp-<version>-offline-<platform>-<abi>.tar.gz
├── corvidex_mcp-<version>+offline-py3-none-any.whl   the package, models embedded
├── wheelhouse/                                       every runtime dependency
├── install.sh                                        venv + pip install --no-index
└── README.md                                         install and verification
```

Measured for 0.1.0 on `manylinux2014_x86_64` / CPython 3.12:

| Artifact | Size |
| --- | --- |
| slim wheel (PyPI) | 1.3 MB |
| sdist (PyPI) | 1.4 MB |
| embedded ONNX weights, uncompressed | 862.3 MB |
| all-models wheel | 537.4 MB |
| dependency wheelhouse (55 wheels) | 71.5 MB |
| **offline bundle archive** | **607.6 MB** |

**This bundle is not on PyPI and cannot be.** PyPI enforces a 100 MB
per-file limit plus a per-project quota, and limit increases are not
granted for bundled model weights. Splitting the weights across several
sub-100 MB wheels to get under the cap was considered and rejected: it
abuses a shared, donated index to host ~0.9 GB of third-party model
binaries. The bundle is published as a **GitHub Release asset** on
<https://github.com/ru551n/corvidex-mcp/releases> instead.

The embedded wheel's version is `<version>+offline` — a PEP 440 local
version segment. PyPI refuses local versions outright, so it cannot be
uploaded there by accident; `pip` still installs it; it sorts *above*
the plain public release, so a host that later gains an index is not
silently downgraded; and `pip show corvidex-mcp` tells you which
variant is installed.

#### Building it (on a connected machine)

```console
$ uv run --no-sync python tools/build_release.py --all
```

That produces both artifacts: `dist/` gets the slim wheel + sdist
(nothing else ever lands there, so `twine check dist/*` is always
safe), and `dist-offline/` gets the bundle archive. Useful flags:

- `--model-source ~/.local/share/corvidex/embed-cache` provisions the
  embedded models by copying an existing fastembed/HF cache instead of
  re-downloading them; add `--models-offline` to make a model missing
  from that cache an error rather than a silent fetch.
- `--python-version` / `--platform` target the wheelhouse at the
  air-gapped host's interpreter and platform (defaults: CPython 3.12,
  the x86_64 manylinux tag set). pip matches `--platform` literally, so
  the default asks for several manylinux variants; the resulting floor
  is the strictest tag actually chosen (today glibc 2.34 — RHEL 9 /
  Ubuntu 22.04 and newer).
- `--slim` / `--offline` build just one of the two.

To provision the models into a working checkout without building
anything (e.g. to develop against the bundled-asset code path):

```console
$ uv run --no-sync python tools/bundle_model.py --list   # what is needed
$ uv run --no-sync python tools/bundle_model.py --all
$ uv run --no-sync python tools/bundle_model.py --all \
      --from ~/.local/share/corvidex/embed-cache --offline
```

`--all` reads the model names from the live `EmbeddingsConfig`
defaults, so it follows a changed default automatically. A single
`--model NAME` still provisions one model, and `--from` accepts either
a Hugging Face cache root or one model's snapshot directory. The ONNX
weights are gitignored; the small JSON files are committed.

#### Installing it (on the air-gapped host)

```console
$ tar xzf corvidex-mcp-0.1.0-offline-manylinux2014-x86-64-cp312.tar.gz
$ cd corvidex-mcp-0.1.0-offline-manylinux2014-x86-64-cp312
$ ./install.sh /opt/corvidex-venv          # PYTHON=... to pick an interpreter
$ claude mcp add corvidex-mcp -- /opt/corvidex-venv/bin/corvidex-mcp
```

#### Verifying it

```console
$ /opt/corvidex-venv/bin/python -m corvidex_mcp.verify_offline
```

The verifier refuses every outbound socket for the rest of the process,
then reports which models resolved to the bundled package assets,
indexes a throwaway repository through the real pipeline (real
chunking, real ONNX embedding, real sqlite-vec + FTS5 store), searches
all three collections, and asserts the cross-encoder reranker actually
ran rather than degrading silently. A download attempt fails loudly
instead of hiding. It ends with `offline verification: OK`.

At runtime the startup self-check logs that each model was loaded from
the bundled assets, and `repository_status` reports per-collection
model state. If a wheel was built *without* the model assets (a plain
git checkout), the fallback is to pre-provision the fastembed cache:
copy an existing `embed-cache` directory from an online machine into
`<data_dir>/embed-cache` before the first start.

## Startup and background sync

The server serves MCP over stdio immediately; it does not wait for the
initial sync to finish. A background task syncs all repositories every
`sync_interval` seconds, and local working repositories are
additionally change-checked every `local_sync_interval` seconds by a
fast, read-only poller. A single-instance lock (`data_dir/server.lock`)
prevents two servers from sharing one data directory.

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

While a repository has not completed its first sync yet, or is being
(re)synced, the search tools prepend a `Note: ...` line to their output
naming it — see [README § Still indexing?](../README.md#still-indexing).

## Operations

- **Data directory** (`data_dir`): the SQLite index (`index.sqlite`), the per-repo
  Git working trees (`<name>/`), sync state
  (`state/repositories.json`), the log file (`logs/corvidex-mcp.log`),
  the model cache (`embed-cache`), the dense-vector cache
  (`dense-cache`), and the lock file. Deleting it resets the index.
- **State & retries**: a repository's `indexed_commit` advances only
  after its index update fully succeeded; a failed sync keeps the
  previous commit and the next sync retries the same diff.
  `last_sync_error` is visible via `repository_status`.
- **Removing a repository** from the config: on the next start the
  server detects it in the state file and automatically drops all of
  its chunks and state.
- **Logs**: `stderr` + `logs/corvidex-mcp.log` (rotating, 3×5 MB).
  `log_level = "DEBUG"` for LSP/git/embedding detail.
