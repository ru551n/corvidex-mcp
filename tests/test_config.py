"""Tests for typed configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vhdl_rag_mcp.config import (
    CATEGORY_DEFAULT_PRIORITIES,
    AppConfig,
    ConfigError,
    QdrantConfig,
    RepositoryCategory,
    load_config,
)

FULL_CONFIG = """\
data_dir = "~/vhdl-rag-data"
sync_interval = 60
vhdl_ls_path = "/opt/vhdl_ls/bin/vhdl_ls"
log_level = "DEBUG"

[embeddings]
vhdl_model = "model-a"
docs_model = "model-b"

[qdrant]
mode = "local"
path = "~/qdata"

[[repositories]]
name = "standards"
url = "git@github.com:co/standards.git"
branch = "main"
category = "golden"
priority = 100

[[repositories]]
name = "ip"
url = "git@github.com:co/ip.git"
branch = "develop"
category = "approved"

[[repositories]]
name = "proj"
url = "https://gitlab.example.com/co/proj.git"
category = "legacy"
"""


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "config.toml", write_default=False)
    assert cfg.data_dir == Path("~/.local/share/vhdl-rag")
    assert cfg.sync_interval == 300
    assert cfg.vhdl_ls_path == "vhdl_ls"
    assert cfg.log_level == "INFO"
    assert cfg.repositories == []
    assert cfg.qdrant.mode == "local"
    assert cfg.qdrant.url is None


def test_default_template_written(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "config.toml")
    assert (tmp_path / "config.toml").exists()
    assert cfg.repositories == []  # template repositories are comments


def test_full_config_parsing(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(FULL_CONFIG, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.data_dir == Path("~/vhdl-rag-data").expanduser()
    assert cfg.sync_interval == 60
    assert cfg.vhdl_ls_path == "/opt/vhdl_ls/bin/vhdl_ls"
    assert cfg.log_level == "DEBUG"
    assert cfg.embeddings.vhdl_model == "model-a"
    assert cfg.embeddings.docs_model == "model-b"
    assert cfg.qdrant.path == Path("~/qdata")
    assert [r.name for r in cfg.repositories] == ["standards", "ip", "proj"]
    std, ip, proj = cfg.repositories
    assert std.branch == "main"
    assert std.category is RepositoryCategory.GOLDEN
    assert std.effective_priority == 100
    assert ip.branch == "develop"
    assert (
        ip.effective_priority
        == CATEGORY_DEFAULT_PRIORITIES[RepositoryCategory.APPROVED]
    )
    assert proj.category is RepositoryCategory.LEGACY
    assert proj.effective_priority == 20
    assert cfg.repository("ip").url == "git@github.com:co/ip.git"


def test_data_dir_expanded(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=Path("~/x"))
    assert str(cfg.resolved_data_dir).startswith(str(Path.home()))
    assert cfg.resolved_data_dir.name == "x"
    assert cfg.repos_dir == cfg.resolved_data_dir / "repos"
    assert cfg.state_dir == cfg.resolved_data_dir / "state"
    assert cfg.log_file == cfg.resolved_data_dir / "logs" / "vhdl-rag.log"
    assert cfg.qdrant_local_path == cfg.resolved_data_dir / "qdrant"


@pytest.mark.parametrize(
    "patch",
    [
        "category = 'not-a-category'\n",
        "sync_interval = 5\n",
        "log_level = 'LOUD'\n",
        "name = 'has space'\n",
    ],
)
def test_invalid_values_rejected(tmp_path: Path, patch: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(FULL_CONFIG.replace('name = "standards"', patch), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_repository_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """\
[[repositories]]
name = "same"
url = "git@github.com:co/a.git"

[[repositories]]
name = "same"
url = "git@github.com:co/b.git"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate repository name"):
        load_config(path)


def test_malformed_toml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("data_dir = [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read configuration"):
        load_config(path)


def test_qdrant_server_mode_requires_url() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"qdrant.url is required"):
        QdrantConfig(mode="server", url=None)
    assert QdrantConfig(mode="server", url="http://qdrant:6333").mode == "server"


def test_repository_lookup_error() -> None:
    cfg = AppConfig()
    with pytest.raises(ConfigError, match="unknown repository"):
        cfg.repository("nope")
