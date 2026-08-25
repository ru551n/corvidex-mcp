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


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be loaded or validated."""


class QdrantConfig(BaseModel):
    """Qdrant connection settings.

    Local (embedded) mode is the default: no separate Qdrant server is
    required; data lives under ``<data_dir>/qdrant``. Server mode is
    supported for the future; only this module and the vector-store
    layer know about Qdrant.
    """

    model_config = ConfigDict(frozen=True)

    mode: str = Field(default="local", pattern="^(local|server)$")
    #: Server-mode endpoint, e.g. ``http://qdrant:6333``.
    url: str | None = None

    @model_validator(mode="after")
    def _check_mode(self) -> QdrantConfig:
        if self.mode == "server" and not self.url:
            raise ValueError('qdrant.url is required when qdrant.mode = "server"')
        return self


class RepositoryConfig(BaseModel):
    """One configured source repository.

    A repository is either a **remote** (``url`` — cloned and kept in
    sync by this server) or a **local working repository** (``path`` —
    the user's own checkout; indexed in place, never mutated). Exactly
    one of the two must be set.

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
    files) are indexed, attributed to the current HEAD commit.

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
            "veridian.yaml when it is missing (invoked before the "
            "Veridian session starts). Same precedence and ownership "
            "semantics as vhdl_ls_hook."
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
        return self

    @property
    def is_local(self) -> bool:
        """True for a local working repository (``path``; no cloning)."""
        return self.path is not None

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
    vhdl_ls_path: str = Field(
        default="vhdl_ls", description="Path to the vhdl_ls binary (or a PATH name)"
    )
    veridian_path: str = Field(
        default="veridian",
        description="Path to the Veridian binary (or a PATH name); Verilog/"
        "SystemVerilog fall back to structural parsing when unavailable",
    )
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    repositories: list[RepositoryConfig] = Field(default_factory=list)

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return Path(value).expanduser()

    @model_validator(mode="after")
    def _unique_repository_names(self) -> AppConfig:
        seen: set[str] = set()
        for repo in self.repositories:
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
    def qdrant_local_path(self) -> Path:
        return self.resolved_data_dir / "qdrant"

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
# remote Git URL (cloned and synced by the server) or a local working
# repository directory (path): the user's own checkout, indexed in place
# without ever being modified. Private remotes work with your normal
# Git/SSH setup (SSH agent, ~/.ssh/config, deploy keys).
#
# "ref" is any resolvable Git ref: a branch name (tracked on every sync),
# a tag, or a commit SHA (full or abbreviated). Tags and SHAs pin the
# repository to a fixed version — sync only verifies the pin. "ref" is
# ignored for local working repositories (path): HEAD plus uncommitted
# changes and untracked files are indexed.
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
vhdl_ls_path = "vhdl_ls"
veridian_path = "veridian"
log_level = "INFO"

# [qdrant]
# mode = "local"
# # or, for a remote Qdrant server:
# # mode = "server"
# # url = "http://qdrant:6333"

[[repositories]]
name = "company-standards"
url = "git@github.com:company/vhdl-standards.git"
ref = "main"               # branch (tracked), tag, or commit SHA (pinned)
# domains = ["hdl", "docs", "code"]   # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]  # glob-style path excludes
# vhdl_ls_hook = "make vhdl-ls-config"  # command to generate vhdl_ls.toml
# veridian_hook = "make veridian-config"  # command to generate veridian.yaml

[[repositories]]
name = "common-ip"
url = "git@github.com:company/common-ip.git"
ref = "v2.1"               # pinned to a release tag

[[repositories]]
name = "current-project"
# url = "git@github.com:company/current-project.git"  # remote (default ref: main)
# ... or index the active checkout instead (never modified by the server):
path = "~/work/current-project"
# ref = "main"               # tracked branch, tag, or SHA (pinned);
#                             # ignored for path repositories
# domains = ["hdl", "docs", "code"]   # which domains to index (default: all)
# exclude = ["sim", "build/*", "*.log"]  # glob-style path excludes
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
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"invalid configuration {config_path}: {exc}") from exc
