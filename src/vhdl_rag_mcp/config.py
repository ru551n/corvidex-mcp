"""Typed TOML configuration for vhdl-rag-mcp.

The configuration lives at ``~/.config/vhdl-rag/config.toml`` by
default; the ``VHDL_RAG_MCP_CONFIG`` environment variable or the
``--config`` command-line flag select an alternate file. On first start
a commented default template is written to the default location if no
file exists. All validation happens at load time; callers receive
either a valid :class:`AppConfig` or a :class:`ConfigError` with an
actionable message.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import CollectionName

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Git refs: branch/tag names or commit SHAs (4-40 hex chars).
REF_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._/-]{0,255}|[0-9a-f]{4,40})$")
#: A full commit SHA: sync can skip the network fetch entirely.
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Every indexable domain; the default per-repository selection.
ALL_DOMAINS: tuple[CollectionName, ...] = (
    CollectionName.HDL,
    CollectionName.DOCS,
    CollectionName.CODE,
)

#: Reserved repository name: the configured coding-standards file is
#: indexed as this pseudo-repository (see ``coding_standards``).
CODING_STANDARDS_REPO = "coding-standards"

#: Accepted extensions for the coding-standards file.
STANDARDS_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".rst", ".pdf", ".docx"}
)


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be loaded or validated."""


class EmbeddingsConfig(BaseModel):
    """Runtime bounds for dense embedding inference (memory safety).

    The local dense models run through ONNX Runtime with the model's full
    context (8192 tokens for the jina v2 models). Attention work is
    quadratic in sequence length, and ONNX Runtime's per-thread memory
    arenas retain peak tensor sizes without releasing them — a single
    long passage in a batch can make one embedding call reserve tens of
    GB. These bounds keep each embedding batch small no matter how long
    the indexed text is.
    """

    model_config = ConfigDict(frozen=True)

    dense_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description=(
            "Maximum tokens a passage is truncated to before dense "
            "embedding. The model's full context is 8192 tokens, but "
            "longer passages rarely improve retrieval while their "
            "attention work (and ONNX Runtime arena usage) grows "
            "quadratically with length."
        ),
    )
    dense_threads: int = Field(
        default=4,
        ge=1,
        description=(
            "Intra-op threads for dense ONNX inference. Lower values "
            "trade per-batch speed for lower memory pressure and less "
            "CPU saturation during indexing and serving."
        ),
    )
    dense_enable_cpu_mem_arena: bool = Field(
        default=False,
        description=(
            "Enable ONNX Runtime's CPU memory arena. When enabled, ONNX "
            "reuses and retains peak tensor buffers across batches, which "
            "is faster but keeps the worst-case arena size resident "
            "(measured ~2.5 GB extra at the defaults). When disabled "
            "(default) buffers are released after each inference, halving "
            "peak RSS at a ~35% indexing-time cost. Disable for "
            "memory-constrained hosts; enable when indexing speed "
            "matters more and RAM is plentiful."
        ),
    )
    dense_batch_size: int = Field(
        default=1,
        ge=1,
        le=128,
        description=(
            "Passages per ONNX inference call during embedding. 1 "
            "(default) bounds each inference to a single truncated "
            "passage, so peak inference memory never grows with batch "
            "content; higher values trade memory for throughput "
            "(padding to the longest passage in the batch, and ONNX "
            "Runtime arena retention, scale with the batch)."
        ),
    )
    index_max_tokens: int = Field(
        default=512,
        ge=64,
        le=8192,
        description=(
            "Maximum tokens an indexed passage is truncated to before "
            "dense embedding. Queries are unaffected (they are short). "
            "Lower values index faster and use less memory; on the "
            "measured corpus quality is unchanged at 512. Must be <= "
            "dense_max_tokens."
        ),
    )
    indexing_workers: int = Field(
        default=1,
        ge=1,
        le=64,
        description=(
            "Worker processes for data-parallel dense embedding during "
            "indexing. 1 (default) = single process. Each worker loads "
            "its own model copy (~0.1 GB for the default small-en model), "
            "so keep it modest and ensure RAM. Speeds up indexing on "
            "multi-core hosts; quality is identical (same model, masked "
            "padding)."
        ),
    )
    hdl_model: str = Field(
        default="jinaai/jina-embeddings-v2-small-en",
        description=(
            "Dense embedding model for the hdl collection (any fastembed "
            "TextEmbedding model name). Default: jina v2 small-en "
            "(512 dims, ~0.12 GB) — measured equal or better retrieval "
            "quality than the base models at ~3x lower indexing RAM. "
            "Switching to a model with a different vector size requires "
            "a reindex (delete the collection or data_dir)."
        ),
    )
    docs_model: str = Field(
        default="jinaai/jina-embeddings-v2-small-en",
        description=(
            "Dense embedding model for the docs collection (any fastembed "
            "TextEmbedding model name; see hdl_model for the default)."
        ),
    )
    code_model: str = Field(
        default="jinaai/jina-embeddings-v2-small-en",
        description=(
            "Dense embedding model for the code collection (any fastembed "
            "TextEmbedding model name; see hdl_model for the default)."
        ),
    )

    @model_validator(mode="after")
    def _check_index_cap(self) -> EmbeddingsConfig:
        if self.index_max_tokens > self.dense_max_tokens:
            raise ValueError(
                "embeddings.index_max_tokens must be <= embeddings.dense_max_tokens"
            )
        return self


class RepositoryConfig(BaseModel):
    """One configured source repository.

    A repository is either a **remote** (``url`` — cloned and kept in
    sync by this server), a **local working repository** (``path`` —
    the user's own checkout; indexed in place, never mutated), or a
    **filesystem repository** (``path`` + ``filesystem = true`` — a plain
    directory of files with no Git at all). Exactly one of ``url`` / ``path``
    must be set.

    For remote repositories, ``ref`` is any resolvable Git ref — a
    branch name, a tag, or a commit SHA (full or abbreviated). A branch
    tracks the remote branch and is updated on every sync; a tag or a
    commit SHA pins the repository, and sync only verifies the pin (a
    full-SHA pin skips the network fetch entirely). Authentication for
    private repositories uses the ambient Git/SSH setup (SSH agent,
    ~/.ssh/config, deploy keys); no credentials are stored by this
    application.

    For local working repositories ``ref`` is ignored: HEAD plus the
    working tree (uncommitted changes and git-respected untracked
    files) are indexed, attributed to the current HEAD commit. Whether
    untracked files are indexed at all is controlled by
    ``index_untracked`` (default: true).

    For filesystem repositories Git is not involved in any way: every
    file below ``path`` is walked and indexed in place (hidden files and
    directories and symlinks are skipped, so an embedded ``.git`` stays
    out of the index). Incremental sync tracks the walked file set plus
    content fingerprints, so edits, additions, and deletions are picked
    up on the next sync. ``ref`` and the Git hooks do not apply.

    ``domains`` selects which of the three indexed domains (HDL,
    documentation, general code) are loaded from this repository; the
    default is all of them. The HDL domain covers VHDL, Verilog, and
    SystemVerilog; the legacy name "vhdl" is accepted as an alias for
    "hdl".
    """

    model_config = ConfigDict(frozen=True)

    name: str
    url: str | None = None
    path: Path | None = None
    ref: str = Field(
        default="main",
        description="Branch, tag, or commit SHA (full or 4-40 hex) to index.",
    )
    domains: list[CollectionName] = Field(
        default_factory=lambda: list(ALL_DOMAINS),
        description=(
            "Domains to index from this repository: any of 'hdl' (VHDL, "
            "Verilog, SystemVerilog; 'vhdl' is accepted as a legacy alias), "
            "'docs', 'code'. Default: all three."
        ),
    )
    exclude: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns (matched against the repository-relative path, "
            "'*' crosses '/') whose files are not indexed; wildcard-free "
            "patterns exclude the whole subtree. E.g. "
            "['sim', 'build/*', '*.log']."
        ),
    )
    vhdl_ls_hook: str | None = Field(
        default=None,
        description=(
            "Shell command, run at the repository root, that produces "
            "vhdl_ls.toml when it is missing (invoked before the vhdl_ls "
            "session starts). When no hook is set, the hook fails, or it "
            "leaves no file behind, the built-in default config is "
            "generated instead. Files the hook creates are owned by the "
            "hook: the server never removes them."
        ),
    )
    veridian_hook: str | None = Field(
        default=None,
        description=(
            "Shell command, run at the repository root, that produces "
            "veridian.yaml when it is missing (invoked before the Veridian "
            "session starts). Same precedence and ownership semantics as "
            "vhdl_ls_hook."
        ),
    )
    filesystem: bool = Field(
        default=False,
        description=(
            "Treat 'path' as a plain directory of files (no Git): every "
            "file is walked and indexed in place, with incremental sync "
            "tracking the file set and content fingerprints. Mutually "
            "exclusive with 'url'; 'ref' and the Git hooks are ignored."
        ),
    )
    index_untracked: bool = Field(
        default=True,
        description=(
            "Local working repositories only: also index untracked files "
            "(honoring .gitignore). No effect on remote repositories and "
            "on filesystem repositories (which always index every file "
            "they find)."
        ),
    )
    priority: int = Field(
        default=1,
        ge=0,
        description=(
            "Retrieval authority weight, applied as a bounded post-RRF "
            "score bonus. 1 (default) adds no bonus; each unit above 1 "
            "ranks this repository's chunks slightly ahead of equally "
            "relevant chunks elsewhere, each unit below 1 slightly "
            "behind. The bonus saturates at a small fraction of the RRF "
            "scale, so it reorders within a relevance tier but can "
            "never promote a chunk across tiers."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not NAME_RE.fullmatch(value):
            raise ValueError(
                "repository name must start with a letter or digit and "
                "contain only [A-Za-z0-9._-]"
            )
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("repository url must not be empty")
        if any(ch.isspace() for ch in value):
            raise ValueError(
                "repository url must not contain whitespace — paste the url "
                "as a single token, e.g. 'https://github.com/org/repo.git' "
                "or 'git@github.com:org/repo.git'"
            )
        return value.strip()

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if not str(value).strip():
            raise ValueError("repository path must not be empty")
        return Path(value).expanduser()

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: str) -> str:
        if not REF_RE.fullmatch(value):
            raise ValueError(
                "repository ref must be a branch/tag name or a commit SHA "
                f"(4-40 hex chars); got {value!r}"
            )
        return value

    @field_validator("domains", mode="before")
    @classmethod
    def _normalize_domains(cls, value: object) -> object:
        # Legacy configs name the HDL domain "vhdl"; accept it as an alias.
        if isinstance(value, list):
            return [
                "hdl" if item in ("vhdl", "hdl", CollectionName.HDL) else item
                for item in value
            ]
        return value

    @field_validator("domains")
    @classmethod
    def _validate_domains(cls, value: list[CollectionName]) -> list[CollectionName]:
        seen: set[CollectionName] = set()
        for domain in value:
            if domain in seen:
                raise ValueError(f"duplicate domain: {domain.value!r}")
            seen.add(domain)
        return value

    @model_validator(mode="after")
    def _check_source(self) -> RepositoryConfig:
        if self.url is None and self.path is None:
            raise ValueError(
                f"repository {self.name!r} must set exactly one of 'url' or 'path'"
            )
        if self.url is not None and self.path is not None:
            raise ValueError(
                f"repository {self.name!r} must set exactly one of 'url' "
                "or 'path', not both"
            )
        if self.filesystem and self.url is not None:
            raise ValueError(
                f"repository {self.name!r}: 'filesystem' requires 'path' "
                "and cannot be combined with 'url'"
            )
        if self.filesystem and self.path is None:
            raise ValueError(f"repository {self.name!r}: 'filesystem' requires 'path'")
        return self

    @property
    def is_local(self) -> bool:
        """True for a local repository (``path``; no cloning): a working
        Git repository or a plain filesystem directory."""
        return self.path is not None

    @property
    def is_filesystem(self) -> bool:
        """True for a plain-directory repository (no Git involved)."""
        return self.filesystem

    @property
    def is_pinned_sha(self) -> bool:
        """True when the ref is a full commit SHA (no fetch needed)."""
        return bool(FULL_SHA_RE.fullmatch(self.ref))

    @property
    def enabled_collections(self) -> frozenset[CollectionName]:
        """The collections this repository contributes to (from ``domains``)."""
        return frozenset(self.domains)


class AppConfig(BaseModel):
    """Validated application configuration."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path = Path("~/.local/share/vhdl-rag")
    sync_interval: int = Field(
        default=300, ge=10, description="Seconds between periodic git synchronizations"
    )
    local_sync_interval: int = Field(
        default=10,
        ge=0,
        le=3600,
        description=(
            "Seconds between change checks of local working repositories "
            "(fast poller; remote repositories still sync on sync_interval). "
            "0 disables the fast poller."
        ),
    )
    vhdl_ls_path: str = Field(
        default="vhdl_ls", description="Path to the vhdl_ls binary (or a PATH name)"
    )
    vhdl_ls_libraries_dir: Path | None = Field(
        default=None,
        description=(
            "Path to vhdl_ls's 'vhdl_libraries' directory (bundled IEEE/std/"
            "synopsys/vital2000 sources). Auto-detected next to the binary "
            "for the official release layout (<root>/bin/vhdl_ls plus "
            "<root>/vhdl_libraries) when unset; that heuristic does not "
            "find anything for a 'cargo install --path' build, in which "
            "case vhdl_ls panics on every invocation ('language server "
            "connection closed' from every sync/reindex) unless this is "
            "set explicitly, e.g. to '<rust_hdl checkout>/vhdl_libraries'."
        ),
    )
    veridian_path: str = Field(
        default="veridian",
        description="Path to the Veridian binary (or a PATH name); Verilog/"
        "SystemVerilog fall back to structural parsing when unavailable",
    )
    coding_standards: Path | None = Field(
        default=None,
        description=(
            "Path to the organization's coding-standards file (txt, md, "
            "rst, pdf, or docx). It is indexed as the pseudo-repository "
            "'coding-standards' in the docs collection with a high "
            "retrieval priority (coding_standards_priority), so it is the "
            "authoritative source for standards/convention questions "
            "within its relevance tier. Absent: no standards file is "
            "indexed."
        ),
    )
    coding_standards_priority: int = Field(
        default=10,
        ge=1,
        description=(
            "Retrieval priority of the coding-standards file "
            "(pseudo-repository 'coding-standards'). 10 (default) is "
            "well above any normal repository priority (1) and applies "
            "the same bounded post-RRF bonus: standards chunks rank "
            "ahead of equally relevant repository chunks, but relevance "
            "still dominates (a standards chunk is not force-promoted "
            "above far more relevant results)."
        ),
    )
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    repositories: list[RepositoryConfig] = Field(default_factory=list)

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return Path(value).expanduser()

    @field_validator("vhdl_ls_libraries_dir")
    @classmethod
    def _expand_vhdl_ls_libraries_dir(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @field_validator("coding_standards")
    @classmethod
    def _validate_coding_standards(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if not str(value).strip():
            raise ValueError("coding_standards must not be empty")
        path = Path(value).expanduser()
        if path.suffix.lower() not in STANDARDS_EXTENSIONS:
            raise ValueError(
                "coding_standards must be a txt, md, rst, pdf, or docx "
                f"file; got {path.suffix or 'no extension'!r}"
            )
        return path

    @model_validator(mode="after")
    def _unique_repository_names(self) -> AppConfig:
        seen: set[str] = set()
        for repo in self.repositories:
            if repo.name == CODING_STANDARDS_REPO:
                raise ValueError(
                    f"repository name {CODING_STANDARDS_REPO!r} is reserved "
                    "for the coding-standards file; choose another name"
                )
            if repo.name in seen:
                raise ValueError(f"duplicate repository name: {repo.name!r}")
            seen.add(repo.name)
        return self

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.resolve()

    @property
    def repos_dir(self) -> Path:
        return self.resolved_data_dir / "repos"

    @property
    def state_dir(self) -> Path:
        return self.resolved_data_dir / "state"

    @property
    def logs_dir(self) -> Path:
        return self.resolved_data_dir / "logs"

    @property
    def embed_cache_dir(self) -> Path:
        return self.resolved_data_dir / "embed-cache"

    @property
    def dense_cache_dir(self) -> Path:
        """Content-addressed cache of dense vectors (see #2)."""
        return self.resolved_data_dir / "dense-cache"

    @property
    def sqlite_index_path(self) -> Path:
        """Embedded SQLite index file (sqlite-vec vec0 + FTS5 tables)."""
        return self.resolved_data_dir / "index.sqlite"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "vhdl-rag.log"

    def repository(self, name: str) -> RepositoryConfig:
        for repo in self.repositories:
            if repo.name == name:
                return repo
        raise ConfigError(
            f"unknown repository {name!r}; configured: "
            f"{', '.join(r.name for r in self.repositories) or '(none)'}"
        )


def default_config_path() -> Path:
    return Path.home() / ".config" / "vhdl-rag" / "config.toml"


_DEFAULT_TEMPLATE = """\
# vhdl-rag-mcp configuration.
#
# Indexed domains: VHDL (via vhdl_ls), VHDL-related documentation, and
# general source code (C/C++, Python, ...). Each repository is either a
# remote Git URL (cloned and synced by the server), a local working
# repository directory (path): the user's own checkout, indexed in place
# without ever being modified, or a plain filesystem directory
# (path + filesystem = true): raw files indexed in place with no Git at
# all. Private remotes work with your normal Git/SSH setup (SSH agent,
# ~/.ssh/config, deploy keys).
#
# "ref" is any resolvable Git ref: a branch name (tracked on every sync),
# a tag, or a commit SHA (full or abbreviated). Tags and SHAs pin the
# repository to a fixed version — sync only verifies the pin. "ref" is
# ignored for local working repositories (path): HEAD plus uncommitted
# changes and untracked files are indexed (untracked indexing can be
# switched off per repository with "index_untracked = false"), and it is
# ignored for filesystem repositories.
#
# "vhdl_ls_hook" / "veridian_hook" are shell commands run at the
# repository root that generate vhdl_ls.toml / veridian.yaml when they
# are missing; without a hook (or if it fails) the server writes a
# built-in default instead.
#
# "domains" selects which domains are indexed from a repository: any
# subset of ["hdl", "docs", "code"]. The hdl domain covers VHDL, Verilog,
# and SystemVerilog ("vhdl" is accepted as a legacy alias). Default: all.
#
# "exclude" lists glob patterns (matched against the repository-relative
# path, '*' crosses '/') whose files are not indexed; wildcard-free
# patterns exclude the whole subtree.
#
# This file can be selected with the VHDL_RAG_MCP_CONFIG environment
# variable or the --config command-line flag. The top-level scalar
# options also have command-line overrides: --data-dir, --sync-interval,
# --vhdl-ls-path, --log-level (command line wins).

data_dir = "~/.local/share/vhdl-rag"
sync_interval = 300
# # Local working repositories (path, not url) are change-checked this
# # often by a fast poller (commits, tracked edits, untracked add/remove);
# # 0 disables it. Remote repositories ignore it (they use sync_interval).
# local_sync_interval = 10
vhdl_ls_path = "vhdl_ls"
veridian_path = "veridian"
log_level = "INFO"
# # The organization's coding standards as one file (txt, md, rst, pdf,
# # or docx): indexed as the 'coding-standards' pseudo-repository with a
# # high retrieval priority (the golden source for conventions):
# coding_standards = "~/standards/coding-standards.md"
# coding_standards_priority = 10

# [embeddings]
# # Max tokens a passage is truncated to before dense embedding (the
# # model's full context is 8192; longer rarely helps and memory cost
# # of attention is quadratic in length):
# dense_max_tokens = 1024
# # CPU threads for dense ONNX inference:
# dense_threads = 4
# # ONNX CPU memory arena (fast but retains peak buffers, ~+2.5 GB;
# # false = lower memory, ~35% slower indexing):
# dense_enable_cpu_mem_arena = false
# # Passages per ONNX inference call (1 = strict per-passage memory
# # bound; higher values trade memory for throughput):
# dense_batch_size = 1
# # Max tokens an indexed passage is truncated to before embedding
# # (queries are unaffected; 512 is measured quality-neutral):
# index_max_tokens = 512
# # Worker processes for data-parallel indexing (1 = single process):
# indexing_workers = 1
# # Per-collection dense model (any fastembed TextEmbedding name;
# # default jina v2 small-en, 512 dims):
# hdl_model = "jinaai/jina-embeddings-v2-small-en"
# docs_model = "jinaai/jina-embeddings-v2-small-en"
# code_model = "jinaai/jina-embeddings-v2-small-en"

# Add one [[repositories]] table per repository to index (uncomment a
# template below or add your own). With none, the server runs with an
# empty index. A repository is a remote Git url (cloned and synced by
# the server) or a local path (your own checkout, indexed in place).
#
# [[repositories]]
# name = "company-standards"
# url = "git@github.com:company/vhdl-standards.git"
# ref = "main"               # branch (tracked), tag, or commit SHA (pinned)
# domains = ["hdl", "docs", "code"]   # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]  # glob-style path excludes
# vhdl_ls_hook = "make vhdl-ls-config"  # command to generate vhdl_ls.toml
# veridian_hook = "make veridian-config"  # command to generate veridian.yaml
#
# [[repositories]]
# name = "common-ip"
# url = "git@github.com:company/common-ip.git"
# ref = "v2.1"               # pinned to a release tag
#
# [[repositories]]
# name = "current-project"
# url = "git@github.com:company/current-project.git"  # remote (default ref: main)
# # ... or index the active checkout instead (never modified by the server):
# path = "~/work/current-project"
# ref = "main"               # tracked branch, tag, or SHA (pinned);
#                             # ignored for path repositories
# domains = ["hdl", "docs", "code"]   # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]  # glob-style path excludes
# index_untracked = false    # skip untracked files (default: index them;
#                             # local working repositories only)

# A plain directory of files, no Git involved: every file below 'path' is
# indexed in place (hidden files/directories and symlinks are skipped, so
# an embedded .git stays out of the index). 'ref' and the Git hooks do not
# apply.
#
# [[repositories]]
# name = "local-ip"
# path = "~/work/local-ip"
# filesystem = true
# domains = ["hdl", "docs"]
"""


def load_config(path: Path | None = None, write_default: bool = True) -> AppConfig:
    """Load and validate the configuration from ``path``.

    When the file does not exist and ``write_default`` is set, a commented
    default template is written and the built-in defaults are returned.
    Raises :class:`ConfigError` on unreadable or invalid configuration.
    """
    if path is None:
        env_path = os.environ.get("VHDL_RAG_MCP_CONFIG")
        path = Path(env_path) if env_path else None
    config_path = (path or default_config_path()).expanduser()
    if not config_path.exists():
        if not write_default:
            return AppConfig()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
        return AppConfig()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        message = f"cannot read configuration {config_path}: {exc}"
        if isinstance(exc, tomllib.TOMLDecodeError):
            # The most common hand-edit mistake is a bare path value
            # (path = ~/git/foo): valid-looking but invalid TOML, and ~
            # only means anything after parsing (the loader expands it).
            message += (
                ". Hint: TOML values must be quoted strings "
                '(e.g. path = "~/git/foo"); a bare ~/... value is not '
                "valid TOML"
            )
        raise ConfigError(message) from exc
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"invalid configuration {config_path}: {exc}") from exc
