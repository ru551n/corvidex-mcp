"""Typed TOML configuration for vhdl-rag-mcp.

The configuration lives at ``~/.config/vhdl-rag/config.toml`` by default.
On first start a commented default template is written there if no file
exists. All validation happens at load time; callers receive either a
valid :class:`AppConfig` or a :class:`ConfigError` with an actionable
message.
"""

from __future__ import annotations

import re
import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be loaded or validated."""


class RepositoryCategory(StrEnum):
    """Authority level of a repository; influences ranking via a bounded bonus."""

    GOLDEN = "golden"
    APPROVED = "approved"
    PROJECT = "project"
    LEGACY = "legacy"


#: Fallback priorities used when a repository does not set ``priority``.
CATEGORY_DEFAULT_PRIORITIES: dict[RepositoryCategory, int] = {
    RepositoryCategory.GOLDEN: 100,
    RepositoryCategory.APPROVED: 90,
    RepositoryCategory.PROJECT: 70,
    RepositoryCategory.LEGACY: 20,
}


class EmbeddingsConfig(BaseModel):
    """Local embedding models per retrieval domain.

    Both collections may use different models; the vector store keeps them
    separate. The jina-embeddings-v2 models are 768-dimensional with an
    8192-token context window, which suits long VHDL constructs.
    """

    model_config = ConfigDict(frozen=True)

    vhdl_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-code",
        description="Code-oriented model used to embed VHDL chunks.",
    )
    docs_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-en",
        description="Text-oriented model used to embed documentation chunks.",
    )


class QdrantConfig(BaseModel):
    """Qdrant connection settings.

    Local (embedded) mode is the default: no separate Qdrant server is
    required. Server mode is supported for the future; only this module
    and the vector-store layer know about Qdrant.
    """

    model_config = ConfigDict(frozen=True)

    mode: str = Field(default="local", pattern="^(local|server)$")
    #: Embedded-mode storage directory; resolved relative to ``data_dir``.
    path: Path | None = None
    #: Server-mode endpoint, e.g. ``http://qdrant:6333``.
    url: str | None = None

    @model_validator(mode="after")
    def _check_mode(self) -> QdrantConfig:
        if self.mode == "server" and not self.url:
            raise ValueError('qdrant.url is required when qdrant.mode = "server"')
        return self


class RepositoryConfig(BaseModel):
    """One configured Git repository.

    Authentication for private repositories uses the ambient Git/SSH setup
    (SSH agent, ``~/.ssh/config``, deploy keys); no credentials are stored
    by this application.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    branch: str = "main"
    category: RepositoryCategory = RepositoryCategory.PROJECT
    priority: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Ranking bonus weight. Defaults to the category default "
            "(golden=100, approved=90, project=70, legacy=20)."
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
    def _validate_url(cls, value: str) -> str:
        if " " in value or not value.strip():
            raise ValueError("repository url must not contain whitespace")
        return value.strip()

    @property
    def effective_priority(self) -> int:
        if self.priority is not None:
            return self.priority
        return CATEGORY_DEFAULT_PRIORITIES[self.category]


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
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
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
        if self.qdrant.path is not None:
            return self.qdrant.path.expanduser().resolve()
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
# Repositories are indexed in priority order; private repositories work
# with your normal Git/SSH setup (SSH agent, ~/.ssh/config, deploy keys).

data_dir = "~/.local/share/vhdl-rag"
sync_interval = 300
vhdl_ls_path = "vhdl_ls"
log_level = "INFO"

[embeddings]
vhdl_model = "jinaai/jina-embeddings-v2-base-code"
docs_model = "jinaai/jina-embeddings-v2-base-en"

# [qdrant]
# mode = "local"
# path = "~/.local/share/vhdl-rag/qdrant"
# # or, for a remote Qdrant server:
# # mode = "server"
# # url = "http://qdrant:6333"

[[repositories]]
name = "company-standards"
url = "git@github.com:company/vhdl-standards.git"
branch = "main"
category = "golden"    # golden | approved | project | legacy
priority = 100

[[repositories]]
name = "current-project"
url = "git@github.com:company/current-project.git"
branch = "main"
category = "project"
"""


def load_config(path: Path | None = None, write_default: bool = True) -> AppConfig:
    """Load and validate the configuration from ``path``.

    When the file does not exist and ``write_default`` is set, a commented
    default template is written and the built-in defaults are returned.
    Raises :class:`ConfigError` on unreadable or invalid configuration.
    """
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
