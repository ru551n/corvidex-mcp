"""Tests for the git repository manager (local file:// remotes, offline)."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from vhdl_rag_mcp.config import RepositoryConfig
from vhdl_rag_mcp.git_manager import GitError, GitManager, parse_name_status_z
from vhdl_rag_mcp.routing import classify_file

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A local 'upstream' repository with one commit."""
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-q", "-b", "main")
    (up / "rtl").mkdir()
    (up / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n")
    (up / "docs").mkdir()
    (up / "docs" / "guide.md").write_text("# Guide\n")
    (up / "src").mkdir()
    (up / "src" / "fifo.c").write_text("int fifo_write(void);\n")
    (up / "README.txt").write_text("readme\n")
    git(up, "add", "-A")
    git(up, "commit", "-qm", "first")
    return up


def make_config(remote: Path, ref: str = "main") -> RepositoryConfig:
    return RepositoryConfig(name="repo", url=str(remote), ref=ref)


@pytest.fixture
def manager(tmp_path: Path) -> GitManager:
    return GitManager(tmp_path / "repos")


def run(coro):
    return asyncio.run(coro)


def test_parse_name_status_z():
    # Format verified against real `git diff -z --name-status` output.
    text = "M\0c.c\0R100\0a.vhd\0d/a2.vhd\0D\0d/b.txt\0A\0new.py\0"
    added, deleted = parse_name_status_z(text)
    assert added == ("c.c", "d/a2.vhd", "new.py")
    assert deleted == ("a.vhd", "d/b.txt")


def test_parse_name_status_z_empty():
    assert parse_name_status_z("") == ((), ())


def test_full_sync_first_run(manager: GitManager, remote: Path):
    plan = run(manager.sync(make_config(remote), None))
    assert plan.full
    assert not plan.empty
    assert plan.deleted == ()
    assert sorted(plan.added_or_modified) == [
        "README.txt",
        "docs/guide.md",
        "rtl/fifo.vhd",
        "src/fifo.c",
    ]
    assert plan.commit == git(remote, "rev-parse", "HEAD")


def test_incremental_sync(manager: GitManager, remote: Path):
    first = run(manager.sync(make_config(remote), None))
    # Modify one, add one, delete one, rename one.
    (remote / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n-- v2\n")
    (remote / "src" / "helper.c").write_text("void help(void);\n")
    git(remote, "rm", "-q", "README.txt")
    git(remote, "mv", "src/fifo.c", "src/fifo2.c")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "second")
    plan = run(manager.sync(make_config(remote), first.commit))
    assert not plan.full
    assert sorted(plan.added_or_modified) == [
        "rtl/fifo.vhd",
        "src/fifo2.c",
        "src/helper.c",
    ]
    assert sorted(plan.deleted) == ["README.txt", "src/fifo.c"]


def test_unchanged_ref_is_empty_plan(manager: GitManager, remote: Path):
    first = run(manager.sync(make_config(remote), None))
    plan = run(manager.sync(make_config(remote), first.commit))
    assert plan.empty
    assert plan.commit == first.commit


def test_tag_pin_does_not_move(manager: GitManager, remote: Path):
    git(remote, "tag", "v1")
    v1 = git(remote, "rev-parse", "v1^{commit}")
    cfg = make_config(remote, ref="v1")
    first = run(manager.sync(cfg, None))
    assert first.commit == v1
    # Upstream advances on main; the tag stays put.
    (remote / "new.vhd").write_text("entity x is end;\n")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "advance")
    plan = run(manager.sync(cfg, first.commit))
    assert plan.empty
    assert plan.commit == v1


def test_pinned_sha_never_fetches(manager: GitManager, remote: Path):
    sha = git(remote, "rev-parse", "HEAD")
    cfg = make_config(remote, ref=sha)
    first = run(manager.sync(cfg, None))
    assert first.commit == sha
    # The upstream vanishes entirely: a full-SHA pin must not need it.
    import shutil

    shutil.rmtree(remote)
    plan = run(manager.sync(cfg, first.commit))
    assert plan.empty


def test_unknown_ref_raises(manager: GitManager, remote: Path):
    cfg = make_config(remote, ref="no-such-branch")
    with pytest.raises(GitError, match="does not resolve"):
        run(manager.sync(cfg, None))


def test_missing_object_falls_back_to_full(manager: GitManager, remote: Path):
    first = run(manager.sync(make_config(remote), None))
    plan = run(manager.sync(make_config(remote), "deadbeef" * 5))
    assert plan.full
    assert plan.commit == first.commit


def test_read_file_and_traversal(manager: GitManager, remote: Path):
    cfg = make_config(remote)
    run(manager.sync(cfg, None))
    assert manager.read_file(cfg, "rtl/fifo.vhd") == "entity fifo is end;\n"
    with pytest.raises(GitError, match="invalid repository file path"):
        manager.read_file(cfg, "../etc/passwd")
    with pytest.raises(GitError, match="not found"):
        manager.read_file(cfg, "nope.vhd")


def test_second_sync_reuses_clone(manager: GitManager, remote: Path):
    cfg = make_config(remote)
    run(manager.sync(cfg, None))
    # Re-running must not fail with "directory already exists".
    run(manager.sync(cfg, git(remote, "rev-parse", "HEAD")))


# -- local working repositories (path) ----------------------------------------


@pytest.fixture
def work_repo(tmp_path: Path) -> Path:
    """A local working repository: a checkout the user owns and edits."""
    repo = tmp_path / "work"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "rtl").mkdir()
    (repo / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text("# Guide\n")
    (repo / "src").mkdir()
    (repo / "src" / "fifo.c").write_text("int fifo_write(void);\n")
    (repo / "README.txt").write_text("readme\n")
    (repo / ".gitignore").write_text("build/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "first")
    return repo


def make_local_config(work_repo: Path) -> RepositoryConfig:
    return RepositoryConfig(name="repo", path=work_repo)


def test_local_first_sync_is_full(manager: GitManager, work_repo: Path):
    plan = run(manager.sync(make_local_config(work_repo), None))
    assert plan.full
    assert plan.commit == git(work_repo, "rev-parse", "HEAD")
    assert plan.ref == "main"
    assert sorted(plan.added_or_modified) == [
        ".gitignore",
        "README.txt",
        "docs/guide.md",
        "rtl/fifo.vhd",
        "src/fifo.c",
    ]


def test_local_uncommitted_and_untracked(manager: GitManager, work_repo: Path):
    cfg = make_local_config(work_repo)
    first = run(manager.sync(cfg, None))
    head = git(work_repo, "rev-parse", "HEAD")
    # Uncommitted work: modify one tracked file, add untracked files.
    (work_repo / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n-- v2\n")
    (work_repo / "src" / "helper.c").write_text("void help(void);\n")
    (work_repo / "build").mkdir()
    (work_repo / "build" / "artifact.o").write_text("junk\n")
    plan = run(manager.sync(cfg, first.commit))
    assert not plan.full
    assert plan.commit == head  # nothing committed: HEAD did not move
    assert plan.ref == "main"
    assert sorted(plan.added_or_modified) == ["rtl/fifo.vhd", "src/helper.c"]
    # .gitignore is honored: build/ is untracked but ignored.
    assert not any(p.startswith("build/") for p in plan.added_or_modified)


def test_local_committed_changes_are_incremental(manager: GitManager, work_repo: Path):
    cfg = make_local_config(work_repo)
    first = run(manager.sync(cfg, None))
    (work_repo / "src" / "helper.c").write_text("void help(void);\n")
    git(work_repo, "rm", "-q", "README.txt")
    git(work_repo, "add", "-A")
    git(work_repo, "commit", "-qm", "second")
    plan = run(manager.sync(cfg, first.commit))
    assert not plan.full
    assert plan.commit == git(work_repo, "rev-parse", "HEAD")
    assert sorted(plan.added_or_modified) == ["src/helper.c"]
    assert plan.deleted == ("README.txt",)


def test_local_unchanged_tree_is_empty_plan(manager: GitManager, work_repo: Path):
    cfg = make_local_config(work_repo)
    first = run(manager.sync(cfg, None))
    plan = run(manager.sync(cfg, first.commit))
    assert plan.empty
    assert plan.commit == first.commit


def test_local_amend_without_content_change_is_empty_plan(
    manager: GitManager, work_repo: Path
):
    cfg = make_local_config(work_repo)
    first = run(manager.sync(cfg, None))
    git(work_repo, "commit", "-q", "--amend", "-m", "rewritten")
    plan = run(manager.sync(cfg, first.commit))
    assert plan.empty
    assert plan.commit != first.commit  # SHA moved, content did not


def test_local_missing_last_commit_falls_back_to_full(
    manager: GitManager, work_repo: Path
):
    cfg = make_local_config(work_repo)
    run(manager.sync(cfg, None))
    # The last indexed commit is gone (history rewrite / force push).
    plan = run(manager.sync(cfg, "deadbeef" * 5))
    assert plan.full
    assert plan.commit == git(work_repo, "rev-parse", "HEAD")


def test_local_invalid_path_rejected(manager: GitManager, tmp_path: Path):
    not_git = tmp_path / "not_git"
    not_git.mkdir()
    with pytest.raises(GitError, match="not a Git repository"):
        run(manager.sync(RepositoryConfig(name="repo", path=not_git), None))
    with pytest.raises(GitError, match="does not exist"):
        run(manager.sync(RepositoryConfig(name="repo", path=tmp_path / "gone"), None))


def test_local_read_file_and_tree_untouched(manager: GitManager, work_repo: Path):
    cfg = make_local_config(work_repo)
    run(manager.sync(cfg, None))
    assert manager.read_file(cfg, "rtl/fifo.vhd") == "entity fifo is end;\n"
    # Untracked files are part of the index and readable.
    (work_repo / "new.py").write_text("x = 1\n")
    assert manager.read_file(cfg, "new.py") == "x = 1\n"
    # The user's tree is never cloned into the managed repos directory.
    assert not (manager._repos_dir / "repo").exists()


# -- routing -----------------------------------------------------------------


def test_routing_vhdl():
    kind = classify_file("rtl/fifo.vhd")
    assert kind is not None
    from vhdl_rag_mcp.models import CollectionName, ContentType

    assert kind.content_type is ContentType.SOURCE
    assert kind.collection is CollectionName.VHDL
    assert kind.language == "vhdl"
    assert classify_file("UPPER/FIFO.VHDL").language == "vhdl"
    assert classify_file("X.VHDL") is not None


def test_routing_docs():
    from vhdl_rag_mcp.models import CollectionName, ContentType

    for ext, lang in (
        (".md", "markdown"),
        (".markdown", "markdown"),
        (".rst", "restructuredtext"),
        (".txt", "text"),
    ):
        kind = classify_file(f"docs/file{ext}")
        assert kind is not None
        assert kind.content_type is ContentType.DOCUMENTATION
        assert kind.collection is CollectionName.DOCS
        assert kind.language == lang


def test_routing_code_and_unknown():
    from vhdl_rag_mcp.models import CollectionName, ContentType

    for ext, lang in (
        (".c", "c"),
        (".h", "c"),
        (".cpp", "cpp"),
        (".py", "python"),
    ):
        kind = classify_file(f"src/file{ext}")
        assert kind is not None
        assert kind.content_type is ContentType.CODE
        assert kind.collection is CollectionName.CODE
        assert kind.language == lang
    assert classify_file("Makefile") is None
    assert classify_file("src/script.sh") is None


def test_routing_domains_restriction():
    from vhdl_rag_mcp.models import CollectionName

    domains = frozenset({CollectionName.VHDL})
    assert classify_file("rtl/fifo.vhd", domains=domains) is not None
    assert classify_file("docs/guide.md", domains=domains) is None
    assert classify_file("src/fifo.c", domains=domains) is None
    # unrestricted
    assert classify_file("src/fifo.c") is not None


def test_routing_excludes():
    excludes = ["sim/*", "*.log", "build/sub"]
    assert classify_file("rtl/fifo.vhd", exclude=excludes) is not None
    assert classify_file("sim/run.vhd", exclude=excludes) is None
    assert classify_file("sim/deep/nested/top.vhd", exclude=excludes) is None
    assert classify_file("rtl/debug.log", exclude=excludes) is None
    assert classify_file("src/debug.log", exclude=excludes) is None
    assert classify_file("build/sub/fifo.vhd", exclude=excludes) is None
    # wildcard-free patterns exclude the whole subtree (gitignore-style)
    assert classify_file("build/sub/deep/file.vhd", exclude=excludes) is None
    assert classify_file("build/fifo.vhd", exclude=excludes) is not None
    # exclude wins over domains
    assert classify_file("rtl/fifo.vhd", domains=None, exclude=["*"]) is None
    assert classify_file("rtl/fifo.vhd", exclude=["rtl"]) is None
