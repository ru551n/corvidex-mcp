# Vhdl Rag MCP

## Overview
Use this skill whenever the user needs to search the organization's
HDL knowledge base through the `vhdl-rag-mcp` MCP server: finding
reference implementations (FIFOs, resets, FSMs, AXI), design
standards/conventions, or answering a question that spans docs, RTL,
and test/simulation code.

The `vhdl-rag-mcp` server indexes configured Git repositories into three
hybrid (dense + full-text) search collections: **hdl** (per-construct
chunks), **docs** (per-section chunks), and **code** (per-function/class
chunks). Every result carries exact source attribution (repository,
file, lines, commit) and the identifiers the chunk references
(`references:`), which is the cross-referencing key between domains.

Triggers on: how do we implement X, is there a reference for X, what does our standard say about X, find every use of an identifier, search the knowledge base.

## Setup (once, per machine)

1. Config at `~/.config/vhdl-rag/config.toml` (or the
   `VHDL_RAG_MCP_CONFIG` env var / `--config` flag) — at least one
   `[[repositories]]` entry with `name` and exactly one of `url`
   (remote, cloned by the server) or `path` (your own local working
   checkout, indexed in place and never modified). For `url` repos set
   `ref` (branch/tag/SHA); `ref` is ignored for `path` repos, whose
   working tree (uncommitted changes + git-respected untracked files)
   is indexed. Per repo: `domains = ["vhdl", "docs", "code"]` and
   `exclude = [...]` select what gets indexed.
2. `vhdl_ls` binary on `PATH` (or `vhdl_ls_path` in config) — required
   for repositories containing VHDL; without it those repositories
   fail their sync (visible in `vhdl_rag_repository_status`). Per repo,
   `vhdl_ls_hook` (a command run at the repo root) can generate the
   workspace `vhdl_ls.toml` when it is missing; without a working hook
   the server writes a built-in default.
3. Register the server (Claude Code): `claude mcp add vhdl-rag-mcp -- uvx --from git+ssh://git@github.com/ru551n/vhdl-rag-mcp.git vhdl-rag-mcp`
4. Verify: call `vhdl_rag_repository_status` — each repo should show an indexed
   commit and no `last error`. If empty or stale, call
   `vhdl_rag_sync_repositories` (or `vhdl_rag_reindex_repository` for one repo).

## Workflow

1. **Start wide, then narrow.** First call:
   `vhdl_rag_search_knowledge(query, limit=10)` — one RRF-fused search over all
   three domains. Good queries name the *thing and the intent*:
   "asynchronous reset de-assertion", "AXI4-Lite slave register file",
   "FIFO with almost-full flag".
2. **Use `symbols` for identifier-level precision.** To find everything
   touching an identifier (a port, signal, or C variable):
   `vhdl_rag_search_vhdl("reset", symbols=["rst_n"])` — results are restricted to
   chunks that reference `rst_n`. Combine domains to trace an
   identifier across docs ↔ RTL ↔ testbench.
3. **Read the exact source before copying.** A search result is a
   chunk (a construct/section/function), not the whole file. Use
   `vhdl_rag_get_source(repository, file, start_line, end_line)` with the
   result's source line to fetch the full construct or file, exactly as
   indexed at the result's commit.
4. **Check `vhdl_rag_repository_status` on any anomaly** (empty results, stale
   content): a repo whose ref moved or whose sync errored will show it
   there; `vhdl_rag_sync_repositories` fixes it.

## Query patterns that work well

- Find a pattern: `vhdl_rag_search_vhdl("process that writes on rising clock edge
  with synchronous reset")`
- Find a standard: `vhdl_rag_search_docs("reset naming conventions")`
- Find a reference implementation: `vhdl_rag_search_knowledge("synchronous FIFO
  with gray pointer")`
- Trace an identifier: `vhdl_rag_search_knowledge("fifo write pointer",
  symbols=["wr_ptr"])`
- Pin to one repo: `vhdl_rag_search_vhdl("AXI handshake", repository="common-ip")`

## Notes and limits

- Results are chunks, not whole files: `content` is the construct/
  section itself, self-contained by design; line numbers refer to the
  repository file at the shown commit.
- The `score` line is the hybrid RRF relevance (dense + full-text
  fusion).
- VHDL chunking is LSP-primary: files with syntax errors fall back to a
  structural scan — still indexed, coarser chunks.
- The index updates automatically (default every 300 s); you normally
  never need to sync manually.
- `vhdl_rag_search_*` `limit` defaults to 8 (10 for `vhdl_rag_search_knowledge`); pass
  `limit=20` for broad exploration.