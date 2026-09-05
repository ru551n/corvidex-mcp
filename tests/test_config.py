"""Tests for typed configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from corvidex_mcp.config import (
    ALL_DOMAINS,
    AppConfig,
    ConfigError,
    EmbeddingsConfig,
    RepositoryConfig,
    load_config,
)
from corvidex_mcp.models import CollectionName

_DEFAULT_DENSE_THREADS = max(1, (os.cpu_count() or 8) // 2)

FULL_CONFIG = """\
data_dir = "~/corvidex-data"
sync_interval = 60
vhdl_ls_path = "/opt/vhdl_ls/bin/vhdl_ls"
vhdl_ls_libraries_dir = "~/git/rust_hdl/vhdl_libraries"
log_level = "DEBUG"

[embeddings]
dense_max_tokens = 2048
dense_threads = 8
dense_enable_cpu_mem_arena = true
index_max_tokens = 128
indexing_workers = 2
hdl_model = "fake/hdl"
docs_model = "fake/docs"
code_model = "fake/code"

[[repositories]]
name = "standards"
url = "git@github.com:co/standards.git"
ref = "main"
domains = ["vhdl", "docs"]
exclude = ["sim/*", "*.log"]
vhdl_ls_hook = "scripts/gen.sh"

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
    cfg = load_config(
        tmp_path / "config.toml", write_default=False, cwd=tmp_path / "workdir"
    )
    assert cfg.data_dir == Path("~/.local/share/corvidex")
    assert cfg.sync_interval == 300
    assert cfg.local_sync_interval == 10
    assert cfg.vhdl_ls_path == "vhdl_ls"
    assert cfg.vhdl_ls_libraries_dir is None
    assert cfg.log_level == "INFO"
    assert cfg.index_cwd is True
    # No [[repositories]] configured: the cwd is indexed automatically.
    assert [r.name for r in cfg.repositories] == ["workdir"]
    assert cfg.repositories[0].path == (tmp_path / "workdir").resolve()
    assert cfg.repositories[0].filesystem is True  # no .git there
    assert cfg.embeddings.dense_max_tokens == 1024
    assert cfg.embeddings.dense_threads == _DEFAULT_DENSE_THREADS
    assert cfg.embeddings.dense_enable_cpu_mem_arena is False
    assert cfg.embeddings.index_max_tokens == 512
    assert cfg.embeddings.indexing_workers == 1
    assert cfg.embeddings.hdl_model == "jinaai/jina-embeddings-v2-small-en"
    assert cfg.embeddings.docs_model == "jinaai/jina-embeddings-v2-small-en"
    assert cfg.embeddings.code_model == "jinaai/jina-embeddings-v2-small-en"
    assert cfg.dense_cache_dir == cfg.resolved_data_dir / "dense-cache"


def test_config_env_var_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CORVIDEX_MCP_CONFIG`` wins; ``VHDL_RAG_MCP_CONFIG`` (the deprecated
    name, from the project's former name vhdl-rag-mcp) is a fallback."""
    new_path = tmp_path / "new.toml"
    old_path = tmp_path / "old.toml"
    new_path.write_text('data_dir = "~/new-data"\n', encoding="utf-8")
    old_path.write_text('data_dir = "~/old-data"\n', encoding="utf-8")

    monkeypatch.setenv("VHDL_RAG_MCP_CONFIG", str(old_path))
    cfg = load_config(write_default=False, cwd=tmp_path)
    assert cfg.data_dir == Path("~/old-data").expanduser()

    monkeypatch.setenv("CORVIDEX_MCP_CONFIG", str(new_path))
    cfg = load_config(write_default=False, cwd=tmp_path)
    assert cfg.data_dir == Path("~/new-data").expanduser()


def test_default_template_written(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    cfg = load_config(tmp_path / "config.toml", cwd=workdir)
    assert (tmp_path / "config.toml").exists()
    # Template repositories are comments; the cwd is still indexed by default.
    assert [r.name for r in cfg.repositories] == ["workdir"]


def test_index_cwd_disabled(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("index_cwd = false\n", encoding="utf-8")
    cfg = load_config(path, cwd=tmp_path)
    assert cfg.repositories == []


def test_configured_repositories_suppress_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[[repositories]]\nname = "explicit"\nurl = "git@x:y.git"\n',
        encoding="utf-8",
    )
    cfg = load_config(path, cwd=tmp_path)
    assert [r.name for r in cfg.repositories] == ["explicit"]


def test_default_repository_for_cwd_git(tmp_path: Path) -> None:
    from corvidex_mcp.config import default_repository_for_cwd

    root = tmp_path / "my-repo"
    (root / ".git").mkdir(parents=True)
    repo = default_repository_for_cwd(root)
    assert repo.name == "my-repo"
    assert repo.path == root.resolve()
    assert repo.filesystem is False


def test_default_repository_for_cwd_sanitizes_name(tmp_path: Path) -> None:
    from corvidex_mcp.config import default_repository_for_cwd

    root = tmp_path / "my project!!"
    root.mkdir()
    repo = default_repository_for_cwd(root)
    assert repo.name == "my-project"

    dotdir = tmp_path / ".hidden"
    dotdir.mkdir()
    assert default_repository_for_cwd(dotdir).name == "workspace"


def test_full_config_parsing(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(FULL_CONFIG, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.data_dir == Path("~/corvidex-data").expanduser()
    assert cfg.sync_interval == 60
    assert cfg.vhdl_ls_path == "/opt/vhdl_ls/bin/vhdl_ls"
    assert (
        cfg.vhdl_ls_libraries_dir == Path("~/git/rust_hdl/vhdl_libraries").expanduser()
    )
    assert cfg.log_level == "DEBUG"
    assert cfg.embeddings.dense_max_tokens == 2048
    assert cfg.embeddings.dense_threads == 8
    assert cfg.embeddings.dense_enable_cpu_mem_arena is True
    assert cfg.embeddings.index_max_tokens == 128
    assert cfg.embeddings.indexing_workers == 2
    assert cfg.embeddings.hdl_model == "fake/hdl"
    assert cfg.embeddings.docs_model == "fake/docs"
    assert cfg.embeddings.code_model == "fake/code"
    assert [r.name for r in cfg.repositories] == [
        "standards",
        "ip",
        "pinned",
        "proj",
        "worktree",
    ]
    std, ip, pinned, proj, worktree = cfg.repositories
    assert std.ref == "main"
    assert std.domains == [CollectionName.HDL, CollectionName.DOCS]
    assert std.enabled_collections == frozenset(
        {CollectionName.HDL, CollectionName.DOCS}
    )
    assert std.exclude == ["sim/*", "*.log"]
    assert std.vhdl_ls_hook == "scripts/gen.sh"
    assert ip.vhdl_ls_hook is None
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
    with pytest.raises(ValidationError, match="must not contain whitespace"):
        RepositoryConfig(name="r", url="git@github.com:org/ repo.git")
    with pytest.raises(ValidationError, match="must not contain whitespace"):
        RepositoryConfig(name="r", url="https://github.com/org/repo .git")
    with pytest.raises(ValidationError, match="path must not be empty"):
        RepositoryConfig(name="r", path="   ")
    # Tilde in the path is expanded at validation time.
    tilde = RepositoryConfig(name="r", path="~/projects/x")
    assert tilde.path == Path("~/projects/x").expanduser()


def test_domains_validation() -> None:
    repo = RepositoryConfig(name="r", url="u", domains=["vhdl", "docs"])
    assert repo.domains == [CollectionName.HDL, CollectionName.DOCS]
    with pytest.raises(ValidationError, match="duplicate domain"):
        RepositoryConfig(name="r", url="u", domains=["vhdl", "vhdl"])
    with pytest.raises(ValidationError, match="Input should be"):
        RepositoryConfig(name="r", url="u", domains=["vhdl", "nope"])


def test_dense_batch_size_validation() -> None:
    from corvidex_mcp.config import EmbeddingsConfig

    assert EmbeddingsConfig().dense_batch_size == 1  # strict per-passage bound
    assert EmbeddingsConfig(dense_batch_size=4).dense_batch_size == 4
    with pytest.raises(ValidationError):
        EmbeddingsConfig(dense_batch_size=0)
    with pytest.raises(ValidationError):
        EmbeddingsConfig(dense_batch_size=1024)


def test_priority_validation() -> None:
    assert RepositoryConfig(name="r", url="u").priority == 1
    assert RepositoryConfig(name="r", url="u", priority=3).priority == 3
    assert RepositoryConfig(name="r", url="u", priority=0).priority == 0
    with pytest.raises(ValidationError):
        RepositoryConfig(name="r", url="u", priority=-1)


def test_data_dir_expanded(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=Path("~/x"))
    assert str(cfg.resolved_data_dir).startswith(str(Path.home()))
    assert cfg.resolved_data_dir.name == "x"
    assert cfg.repos_dir == cfg.resolved_data_dir / "repos"
    assert cfg.state_dir == cfg.resolved_data_dir / "state"
    assert cfg.log_file == cfg.resolved_data_dir / "logs" / "corvidex.log"
    assert cfg.sqlite_index_path == cfg.resolved_data_dir / "index.sqlite"


def test_local_sync_interval_bounds(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("local_sync_interval = 0\n")  # 0 disables the poller
    assert load_config(path).local_sync_interval == 0
    path.write_text("local_sync_interval = -1\n")
    with pytest.raises(ConfigError):
        load_config(path)
    path.write_text("local_sync_interval = 3601\n")
    with pytest.raises(ConfigError):
        load_config(path)


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


def test_unquoted_path_gets_quoting_hint(tmp_path: Path) -> None:
    """A bare ~/path (the common hand-edit mistake) is invalid TOML; the
    error must point at the quoting so the user can self-correct."""
    path = tmp_path / "config.toml"
    path.write_text(
        '[[repositories]]\nname = "proj"\npath = ~/git/proj\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"cannot read configuration.*quoted strings"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dense_max_tokens": 63},  # below the 64-token floor
        {"dense_max_tokens": 8193},  # above the model context
        {"dense_threads": 0},  # needs at least one thread
        {"index_max_tokens": 63},  # below the 64-token floor
        {"index_max_tokens": 8193},  # above the model context
        {"indexing_workers": 0},  # needs at least one worker
        # index cap may not exceed the query cap
        {"index_max_tokens": 2048, "dense_max_tokens": 1024},
    ],
)
def test_embeddings_bounds_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        AppConfig(embeddings=EmbeddingsConfig(**kwargs))


def test_repository_lookup_error() -> None:
    cfg = AppConfig()
    with pytest.raises(ConfigError, match="unknown repository"):
        cfg.repository("nope")


def test_analyzer_path_defaults() -> None:
    cfg = AppConfig()
    assert cfg.vhdl_ls_path == "vhdl_ls"
    assert cfg.veridian_path == "veridian"


def test_filesystem_repository_config() -> None:
    repo = RepositoryConfig(
        name="fsrepo", path=Path("~/work/local-ip"), filesystem=True
    )
    assert repo.is_filesystem
    assert repo.is_local
    assert repo.path == Path("~/work/local-ip").expanduser()
    # 'ref' is accepted (default kept) but does not apply.
    assert repo.ref == "main"


def test_filesystem_validation() -> None:
    with pytest.raises(ValidationError, match="'filesystem'"):
        RepositoryConfig(name="r", url="git@x:y.git", filesystem=True)
    with pytest.raises(ValidationError, match="'filesystem'"):
        RepositoryConfig(name="r", filesystem=True)


def test_index_untracked_flag() -> None:
    assert RepositoryConfig(name="r", url="u").index_untracked is True
    assert (
        RepositoryConfig(name="r", url="u", index_untracked=False).index_untracked
        is False
    )


def test_filesystem_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[[repositories]]\n"
        'name = "fsrepo"\n'
        'path = "~/work/local-ip"\n'
        "filesystem = true\n"
        "index_untracked = false\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    repo = cfg.repositories[0]
    assert repo.is_filesystem
    assert repo.index_untracked is False
    assert repo.path == Path("~/work/local-ip").expanduser()


def test_veridian_hook_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'veridian_path = "/opt/veridian/veridian"\n'
        "[[repositories]]\n"
        'name = "r"\n'
        'url = "u"\n'
        'veridian_hook = "make veridian-config"\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.veridian_path == "/opt/veridian/veridian"
    assert cfg.repositories[0].veridian_hook == "make veridian-config"
    # Default is None (no hook) when unset.
    assert RepositoryConfig(name="r", url="u").veridian_hook is None
