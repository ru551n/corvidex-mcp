---
name: corvidex-mcp
description: Semantic search over the organization's VHDL code, coding standards, VHDL-related documentation, and general source code via the corvidex-mcp MCP server (search_hdl, search_vhdl, search_docs, search_code, search_knowledge, get_source, repository_files), plus exact LSP-backed navigation (find_definition, find_references, hover_info, find_symbol). Use when implementing or modifying HDL, looking up or enforcing design standards/conventions (the coding-standards file is the golden source), finding reference implementations (FIFOs, resets, FSMs, AXI), resolving a symbol's exact declaration/use sites, or when a question spans docs, RTL, and test/simulation code.
---

# corvidex-mcp — RTL-centric RAG and indexing MCP

The `corvidex-mcp` server indexes configured Git repositories into three
hybrid (dense + full-text, RRF-fused) search collections: **hdl**
(per-construct chunks), **docs** (per-section chunks), and **code**
(per-function/class chunks). When the config sets `coding_standards`,
the organization's coding-standards file is additionally indexed as the
**`coding-standards` pseudo-repository** (docs collection) with a high
retrieval priority: it is the **golden source** for standards questions.
Every result carries exact source attribution (repository, file, lines,
commit) and the identifiers the chunk references, which is the
cross-referencing key between domains.

## Setup (once, per machine)

1. Config at a project-local `.corvidex` file in the directory the
   server is started in (or the `CORVIDEX_MCP_CONFIG` env var / `--config`
   flag) — one
   `[[repositories]]` entry per repository with `name` and exactly one
   of `url` (remote, cloned by the server) or `path` (your own local
   working checkout, indexed in place and never modified). For `url`
   repos set `ref` (branch/tag/SHA); `ref` is ignored for `path` repos,
   whose working tree (uncommitted changes + git-respected untracked
   files) is indexed. Per repo: `domains = ["hdl", "docs", "code"]` and
   `exclude = [...]` select what gets indexed; `priority` (default 1)
   weights the repo in retrieval.
2. **Coding standards (recommended):** set `coding_standards` to the
   standards file (txt, md, rst, pdf, or docx). It becomes the
   `coding-standards` pseudo-repository (search it with
   `repository="coding-standards"`); edits are picked up on the next
   sync automatically.
3. `vhdl_ls` binary on `PATH` (or `vhdl_ls_path`) for VHDL repositories;
   without it those files fall back to structural parsing (still
   indexed, coarser chunks). Per repo, `vhdl_ls_hook` / `veridian_hook`
   can generate the workspace config when missing.
4. Register the server (Claude Code): `claude mcp add corvidex-mcp -- uvx --from git+ssh://git@github.com/ru551n/corvidex-mcp.git corvidex-mcp`
5. Verify: call `repository_status` — each repo should show an indexed
   commit and no `last error` (a misconfigured url/ref shows an
   actionable hint there). If empty or stale, call `sync_repositories`.

## Workflow

1. **Coding standards and conventions: go to the golden source first.**
   For any standards, convention, naming, or "how do we do X"
   question, call
   `search_docs("...", repository="coding-standards")` (or
   `search_knowledge` if the standard may also live in repo docs).
   Those chunks carry the highest retrieval priority: when the
   standards file covers the topic, its sections outrank repository
   chunks of equal relevance. Treat the golden file as authoritative;
   repository docs are secondary context — prefer the golden source
   when they disagree.
2. **Start wide, then narrow.** `search_knowledge(query, limit=10)` —
   one RRF-fused search over all domains. Good queries name the *thing
   and the intent*: "asynchronous reset de-assertion", "AXI4-Lite
   slave register file", "FIFO with almost-full flag".
3. **Pick the search mode to the question.** Every search tool takes
   `mode` (default `hybrid`):
   - `hybrid` — semantic + full-text fused; the default for most work;
   - `semantic` — embedding similarity only; for "find what this means"
     queries where wording differs from the code;
   - `lexical` — exact full-text match only, no embedding (fastest);
     for identifiers, literal names, and standard terms, e.g.
     `search_docs("rst_n", mode="lexical", repository="coding-standards")`.
4. **Use `symbols` for identifier-level precision.** To find everything
   touching an identifier (a port, signal, or C variable):
   `search_hdl("reset", symbols=["rst_n"])` — results are restricted to
   chunks that reference `rst_n`. Combine domains to trace an
   identifier across docs ↔ RTL ↔ testbench (code snippets in standard
   sections export their identifiers too).
5. **Read the exact source before copying.** A search result is a
   chunk (a construct/section/function), not the whole file, and it is
   quoted at most 40 lines deep with a 1-based line-number gutter. When
   a chunk was longer, the last body line is the exact call for the
   rest — e.g.
   `… 158 more lines — get_source("common-ip", "rtl/fifo.vhd", 635, 792) for the full text`
   — run it verbatim rather than re-deriving the arguments. Otherwise
   use
`get_source(repository, file, start_line, end_line)` with the
    result's source line to fetch the full construct or file, exactly as
    indexed at the result's commit. For the standards file,
    `repository="coding-standards"` works here too. When a file path is
    unknown, don't guess it — list the indexed files with
    `repository_files(repository, pattern)` (a glob on the
    repository-relative path, e.g. `*.vhd` or `modules/<ip>/*`) and pass
    the result to `get_source`.
6. **Check `repository_status` on any anomaly** (empty results, stale
   content, `last error` with a git hint): a repo whose ref moved or
   whose sync errored shows it there; `sync_repositories` (or
   `reindex_repository` for one repo) fixes it.
7. **Need an exact answer, not a similarity match?** Once a search
   result (or an earlier navigation call) gives an exact `file:line`,
   switch to the LSP-backed navigation tools (`vhdl_ls` for VHDL,
   Veridian for Verilog/SystemVerilog) instead of guessing from search
   text: `find_definition(repository, file, line, character)` for the
   declaration site, `find_references(...)` for every use site,
   `hover_info(...)` for the analyzer's own signature/type text, and
   `find_symbol(query, repository?)` for an exact name-based lookup
   across one or every repository. `line`/`character` are 0-based (LSP
   convention); results render as 1-based `path:line:col`. Search
   results and `get_source` print 1-based line numbers in their gutter,
   so a line displayed as `N` is passed to these tools as `N - 1` — read
   the number off the gutter instead of counting from the `source:`
   range.

## Query patterns that work well

- Enforce a standard: `search_docs("reset naming", repository="coding-standards")`
- Enforce a standard, exact term: `search_docs("rst_n", mode="lexical", repository="coding-standards")`
- Find a pattern: `search_hdl("process that writes on rising clock edge with synchronous reset")`
- Find a standard in repo docs: `search_docs("reset naming conventions")`
- Find a reference implementation: `search_knowledge("synchronous FIFO with gray pointer")`
- Trace an identifier: `search_knowledge("fifo write pointer", symbols=["wr_ptr"])`
- Pin to one repo: `search_hdl("AXI handshake", repository="common-ip")`
- VHDL-only: `search_vhdl("architecture with clocked process")`
- Exact declaration from a known position: `find_definition("common-ip", "rtl/fifo.vhd", 41, 12)`
- Every use site of a known symbol: `find_references("common-ip", "rtl/fifo.vhd", 12, 6)`
- Exact name lookup: `find_symbol("wr_ptr", repository="common-ip")`

## Notes and limits

- Results are chunks, not whole files: `content` is the
  construct/section itself, self-contained by design; the gutter's
  1-based line numbers refer to the repository file at the shown
  commit. Only the first 40 lines of a chunk are quoted — the head,
  where entity headers, port lists and process declarations live — and
  an elided body ends with the `get_source` call for the remainder.
- A chunk nested inside a better-ranked chunk of the same file (a
  process inside its architecture) is dropped from the response, so
  the same lines are never quoted twice.
- The `score` line is the cross-encoder reranker's relevance in `0..1`
  when reranking is available (comparable across queries), otherwise
  the hybrid RRF/cosine score, which is only comparable within one
  response; repository priority (and the high coding-standards
  priority) adds a small bounded bonus that reorders within a relevance
  tier but never promotes a weak chunk above strong matches. A
  `Note: no strong match` first line means the best reranked hit scored
  below ~0.05 — take it at face value: use `find_symbol` for an exact
  name, or rephrase, instead of mining the listed hits.
- A "more matches exist beyond `limit`" note is appended only when
  further candidates really existed; no note means the response is
  everything the index has for that query.
- The index updates automatically (default every 300 s; local working
  repos every 10 s); you normally never need to sync manually.
- `search_*` `limit` defaults to 8 (10 for `search_knowledge`).
- Search modes: `lexical` never loads the embedding model (works when
  the model is unavailable); `semantic`/`hybrid` need it.
