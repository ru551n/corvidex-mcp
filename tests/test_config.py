"""Tests for typed configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vhdl_rag_mcp.config import (
    ALL_DOMAINS,
    AppConfig,
    ConfigError,
    QdrantConfig,
    RepositoryConfig,
    load_config,
)
from vhdl_rag_mcp.models import CollectionName

FULL_CONFIG = """\
data_dir = "~/vhdl-rag-data"
sync_interval = 60
vhdl_ls_path = "/opt/vhdl_ls/bin/vhdl_ls"
log_level = "DEBUG"

[qdrant]
mode = "local"

[[repositories]]
name = "standards"
url = "git@github.com:co/standards.git"
ref = "main"
domains = ["vhdl", "docs"]
exclude = ["sim/*", "*.log"]

[[repositories]]
name = "ip"
url = "git@github.com:co/ip.git"
ref = "v2.1"

[[repositories]]
name = "pinned"
url = "git@github.com:co/pinned.git"
ref = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"

[[repositories]]
name = "proj"
url = "https://gitlab.example.com/co/proj.git"
domains = ["code"]

[[repositories]]
name = "worktree"
path = "~/work/current-project"
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
    assert cfg.qdrant.mode == "local"
    assert [r.name for r in cfg.repositories] == [
        "standards",
        "ip",
        "pinned",
        "proj",
        "worktree",
    ]
    std, ip, pinned, proj, worktree = cfg.repositories
    assert std.ref == "main"
    assert std.domains == [CollectionName.VHDL, CollectionName.DOCS]
    assert std.enabled_collections == frozenset(
        {CollectionName.VHDL, CollectionName.DOCS}
    )
    assert std.exclude == ["sim/*", "*.log"]
    assert ip.ref == "v2.1"
    # domains default to all three; exclude defaults to empty
    assert ip.domains == list(ALL_DOMAINS)
    assert ip.enabled_collections == frozenset(ALL_DOMAINS)
    assert ip.exclude == []
    assert pinned.ref == "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
    assert pinned.is_pinned_sha
    assert not std.is_pinned_sha
    assert not ip.is_pinned_sha
    assert proj.domains == [CollectionName.CODE]
    assert cfg.repository("ip").url == "git@github.com:co/ip.git"
    # Local working repository: path instead of url, ref defaults away.
    assert worktree.url is None
    assert worktree.path == Path("~/work/current-project").expanduser()
    assert worktree.is_local
    assert not std.is_local and not ip.is_local


def test_repository_url_xor_path() -> None:
    remote = RepositoryConfig(name="r", url="u")
    local = RepositoryConfig(name="r", path=Path("/tmp/work"))
    assert remote.path is None and not remote.is_local
    assert local.url is None and local.is_local
    assert local.path == Path("/tmp/work")
    with pytest.raises(ValidationError, match="exactly one of 'url' or 'path'"):
        RepositoryConfig(name="r", url="u", path=Path("/tmp/work"))
    with pytest.raises(ValidationError, match="exactly one of 'url' or 'path'"):
        RepositoryConfig(name="r")
    with pytest.raises(ValidationError, match="url must not be empty"):
        RepositoryConfig(name="r", url="   ")
    with pytest.raises(ValidationError, match="path must not be empty"):
        RepositoryConfig(name="r", path="   ")
    # Tilde in the path is expanded at validation time.
    tilde = RepositoryConfig(name="r", path="~/projects/x")
    assert tilde.path == Path("~/projects/x").expanduser()


def test_domains_validation() -> None:
    repo = RepositoryConfig(name="r", url="u", domains=["vhdl", "docs"])
    assert repo.domains == [CollectionName.VHDL, CollectionName.DOCS]
    with pytest.raises(ValidationError, match="duplicate domain"):
        RepositoryConfig(name="r", url="u", domains=["vhdl", "vhdl"])
    with pytest.raises(ValidationError, match="Input should be"):
        RepositoryConfig(name="r", url="u", domains=["vhdl", "nope"])


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
        "sync_interval = 5\n",
        "log_level = 'LOUD'\n",
        "name = 'has space'\n",
        'ref = "main..dev"\n',  # not a resolvable ref syntax
        'ref = "abc"\n',  # SHA too short
    ],
)
def test_invalid_values_rejected(tmp_path: Path, patch: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(FULL_CONFIG.replace('name = "standards"', patch), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_ref_accepts_branch_tag_and_sha_prefix() -> None:
    cfg = AppConfig(
        repositories=[
            {"name": "a", "url": "u", "ref": "main"},
            {"name": "b", "url": "u", "ref": "release/2024-01"},
            {"name": "c", "url": "u", "ref": "abc12345"},
            {"name": "d", "url": "u"},
        ]
    )
    refs = [r.ref for r in cfg.repositories]
    assert refs == ["main", "release/2024-01", "abc12345", "main"]
    assert [r.is_pinned_sha for r in cfg.repositories] == [False, False, False, False]
    assert cfg.repositories[2].ref == "abc12345"


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
    with pytest.raises(ValidationError, match=r"qdrant.url is required"):
        QdrantConfig(mode="server", url=None)
    assert QdrantConfig(mode="server", url="http://qdrant:6333").mode == "server"


def test_repository_lookup_error() -> None:
    cfg = AppConfig()
    with pytest.raises(ConfigError, match="unknown repository"):
        cfg.repository("nope")
