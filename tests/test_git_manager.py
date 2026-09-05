"""Tests for the git repository manager (local file:// remotes, offline)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from corvidex_mcp.config import RepositoryConfig
from corvidex_mcp.git_manager import (
    GitError,
    GitManager,
    _hint,
    parse_name_status_z,
)
from corvidex_mcp.routing import classify_file

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


def git_live(cwd: Path, *args: str) -> str:
    """Like :func:`git`, but builds the environment from the current
    ``os.environ`` (so ``protocol.file.allow`` set by the ``sub_env``
    fixture reaches the git subprocess)."""
    env = dict(os.environ)
    for key in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        env.setdefault(key, ENV[key])
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def sub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow the file:// protocol (test submodules use local paths)."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")


def make_upstream_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    """A superproject with one submodule ``ip``; returns (up, sub_up)."""
    sub_up = tmp_path / "sub-upstream"
    sub_up.mkdir()
    git_live(sub_up, "init", "-q", "-b", "main")
    (sub_up / "rtl").mkdir()
    (sub_up / "rtl" / "a.vhd").write_text("entity a is end;\n")
    git_live(sub_up, "add", "-A")
    git_live(sub_up, "commit", "-qm", "sub first")

    up = tmp_path / "upstream"
    up.mkdir()
    git_live(up, "init", "-q", "-b", "main")
    (up / "top.vhd").write_text("entity top is end;\n")
    git_live(up, "add", "-A")
    git_live(up, "commit", "-qm", "super first")
    git_live(up, "submodule", "add", "-q", str(sub_up), "ip")
    git_live(up, "commit", "-qm", "add submodule")
    return up, sub_up


@pytest.fixture
def remote_with_submodule(tmp_path: Path, sub_env: None) -> tuple[Path, Path]:
    return make_upstream_with_submodule(tmp_path)


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


def rmtree_with_retry(path: Path, attempts: int = 12) -> None:
    """``shutil.rmtree`` that retries through transient Windows locks.

    Files created seconds ago can still be briefly locked on Windows
    (e.g. Defender real-time scanning), where a plain rmtree raises
    PermissionError. If the lock outlives the retries, fall back to
    best effort: the assertion that follows does not depend on the
    deletion (the pinned-SHA code path never touches the remote,
    whether it exists or not).
    """
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.5)
    shutil.rmtree(path, ignore_errors=True)


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
    rmtree_with_retry(remote)
    plan = run(manager.sync(cfg, first.commit))
    assert plan.empty


def test_unknown_ref_raises(manager: GitManager, remote: Path):
    cfg = make_config(remote, ref="no-such-branch")
    with pytest.raises(GitError, match="does not resolve"):
        run(manager.sync(cfg, None))


def test_unknown_ref_lists_available_refs(manager: GitManager, remote: Path) -> None:
    cfg = make_config(remote, ref="no-such-branch")
    with pytest.raises(GitError, match="does not resolve to a commit") as excinfo:
        run(manager.sync(cfg, None))
    # Actionable: the available remote refs are listed.
    assert "available remote refs" in str(excinfo.value)
    assert "origin/main" in str(excinfo.value)


def test_non_repository_url_error_has_hint(manager: GitManager, tmp_path: Path) -> None:
    # An existing directory that is not a git repository.
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    cfg = RepositoryConfig(name="repo", url=str(not_a_repo), ref="main")
    with pytest.raises(GitError) as excinfo:
        run(manager.sync(cfg, None))
    message = str(excinfo.value)
    assert "hint:" in message
    assert "reachable git repository" in message


def test_hint_classification() -> None:
    assert "reachable git repository" in _hint(
        "fatal: '/x' does not appear to be a git repository."
    )
    assert "reachable git repository" in _hint(
        "fatal: repository '/x/not-a-repo' does not exist"
    )
    assert "name/spelling" in _hint(
        "fatal: repository 'https://github.com/x/y/' not found."
    )
    assert "resolved" in _hint(
        "ssh: Could not resolve hostname github.com: Temporary failure"
    )
    assert "known_hosts" in _hint("Host key verification failed.")
    assert "ssh" in _hint("git@github.com: Permission denied (publickey).")
    assert "token" in _hint(
        "fatal: Authentication failed for 'https://github.com/x/y.git/'"
    )
    assert "timed out" in _hint("Connection timed out after 30 seconds")
    assert "proxy" in _hint(
        "fatal: unable to access 'https://x/': Could not resolve proxy"
    )
    assert _hint("fatal: some other obscure failure") == ""


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
    with pytest.raises(GitError, match=r"not found.*repository_files"):
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
    assert first.fingerprint is not None
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
    # Tracked edits land in added_or_modified; untracked files are reported
    # separately for content-fingerprinting by the pipeline.
    assert sorted(plan.added_or_modified) == ["rtl/fifo.vhd"]
    assert plan.untracked == ("src/helper.c",)
    # .gitignore is honored: build/ is untracked but ignored.
    assert not any(p.startswith("build/") for p in plan.untracked)
    assert plan.fingerprint != first.fingerprint


def test_local_fingerprint_tracks_working_tree(manager: GitManager, work_repo: Path):
    cfg = make_local_config(work_repo)
    fp = run(manager.local_fingerprint(cfg))
    # Stable while the tree is untouched.
    assert run(manager.local_fingerprint(cfg)) == fp
    # Tracked edit (unstaged).
    (work_repo / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n-- v2\n")
    fp2 = run(manager.local_fingerprint(cfg))
    assert fp2 != fp
    # Commit: HEAD moves.
    git(work_repo, "add", "-A")
    git(work_repo, "commit", "-qm", "v2")
    fp3 = run(manager.local_fingerprint(cfg))
    assert fp3 != fp2
    # Untracked file appears.
    (work_repo / "src" / "helper.c").write_text("void help(void);\n")
    fp4 = run(manager.local_fingerprint(cfg))
    assert fp4 != fp3
    # Untracked file removed.
    (work_repo / "src" / "helper.c").unlink()
    assert run(manager.local_fingerprint(cfg)) == fp3
    # An edit to an *existing* untracked file does not change it (the
    # pipeline's content fingerprinting covers that case instead).
    (work_repo / "src" / "helper.c").write_text("void help(void);\n")
    fp5 = run(manager.local_fingerprint(cfg))
    (work_repo / "src" / "helper.c").write_text("void help2(void);\n")
    assert run(manager.local_fingerprint(cfg)) == fp5


def test_local_untracked_removal_is_empty_plan_for_pipeline(
    manager: GitManager, work_repo: Path
):
    """The manager keeps reporting untracked files even when nothing else
    changed; the pipeline (not the manager) decides what to do with them."""
    cfg = make_local_config(work_repo)
    first = run(manager.sync(cfg, None))
    (work_repo / "src" / "helper.c").write_text("void help(void);\n")
    plan = run(manager.sync(cfg, first.commit))
    assert plan.added_or_modified == ()
    assert plan.deleted == ()
    assert plan.untracked == ("src/helper.c",)
    # Fingerprint changed (a new untracked file appeared), so the fast
    # poller will trigger this sync.
    assert plan.fingerprint is not None
    assert plan.fingerprint != first.fingerprint


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
    from corvidex_mcp.models import CollectionName, ContentType

    assert kind.content_type is ContentType.SOURCE
    assert kind.collection is CollectionName.HDL
    assert kind.language == "vhdl"
    assert classify_file("UPPER/FIFO.VHDL").language == "vhdl"
    assert classify_file("X.VHDL") is not None


def test_routing_verilog():
    from corvidex_mcp.models import CollectionName, ContentType

    for path, lang in (
        ("rtl/fifo.v", "verilog"),
        ("rtl/pkg.sv", "systemverilog"),
        ("inc/defs.svh", "systemverilog"),
    ):
        kind = classify_file(path)
        assert kind is not None
        assert kind.content_type is ContentType.SOURCE
        assert kind.collection is CollectionName.HDL
        assert kind.language == lang
    # The HDL domain admits all three languages together...
    hdl_only = frozenset({CollectionName.HDL})
    assert classify_file("rtl/fifo.v", domains=hdl_only) is not None
    assert classify_file("rtl/fifo.sv", domains=hdl_only) is not None
    assert classify_file("rtl/fifo.vhd", domains=hdl_only) is not None
    # ...while a code-only selection admits none of them.
    code_only = frozenset({CollectionName.CODE})
    assert classify_file("rtl/fifo.v", domains=code_only) is None
    # Upper-case extensions route the same.
    assert classify_file("RTL/FIFO.SV").language == "systemverilog"


def test_routing_docs():
    from corvidex_mcp.models import CollectionName, ContentType

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
    from corvidex_mcp.models import CollectionName, ContentType

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
    from corvidex_mcp.models import CollectionName

    domains = frozenset({CollectionName.HDL})
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


# -- filesystem repositories ---------------------------------------------------


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    root = tmp_path / "fswalk"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl" / "a.vhd").write_text("entity a is end;\n")
    (root / "rtl" / "b.vhd").write_text("entity b is end;\n")
    (root / "doc").mkdir()
    (root / "doc" / "guide.md").write_text("# Guide\n")
    (root / "deep" / "nested").mkdir(parents=True)
    (root / "deep" / "nested" / "c.vhd").write_text("entity c is end;\n")
    # Must be skipped by the walk: hidden dir, hidden file, symlink.
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".hidden.vhd").write_text("entity hidden is end;\n")
    (root / "link.vhd").symlink_to(root / "rtl" / "a.vhd")
    return root


def fs_config(root: Path) -> RepositoryConfig:
    return RepositoryConfig(name="fsrepo", path=root, filesystem=True)


def test_filesystem_first_sync(manager: GitManager, fs_root: Path):
    plan = run(manager.sync(fs_config(fs_root), None))
    assert plan.full
    assert not plan.empty
    assert plan.deleted == ()
    assert plan.ref == "-"
    # Attribution is the walk fingerprint itself.
    assert plan.commit == plan.fingerprint
    assert sorted(plan.added_or_modified) == [
        "deep/nested/c.vhd",
        "doc/guide.md",
        "rtl/a.vhd",
        "rtl/b.vhd",
    ]
    # The first sync also carries the file set so the pipeline can
    # fingerprint its content.
    assert sorted(plan.untracked) == sorted(plan.added_or_modified)
    # Hidden entries and symlinks never leak into the plan.
    assert all(not p.startswith(".git") for p in plan.untracked)
    assert ".hidden.vhd" not in plan.untracked
    assert "link.vhd" not in plan.untracked


def test_filesystem_unchanged_is_empty(manager: GitManager, fs_root: Path):
    cfg = fs_config(fs_root)
    first = run(manager.sync(cfg, None))
    second = run(manager.sync(cfg, first.commit))
    assert not second.full
    assert second.empty
    assert second.commit == first.commit
    assert sorted(second.untracked) == sorted(first.untracked)


def test_filesystem_walk_tracks_add_and_remove(manager: GitManager, fs_root: Path):
    cfg = fs_config(fs_root)
    first = run(manager.sync(cfg, None))

    (fs_root / "rtl" / "new.vhd").write_text("entity new is end;\n")
    (fs_root / "doc" / "guide.md").unlink()
    second = run(manager.sync(cfg, first.commit))
    # No tracked-level change, so the plan is 'empty' in the SyncPlan
    # sense; the pipeline's untracked refinement picks up the file-set
    # change from ``untracked`` + the moved fingerprint.
    assert second.empty
    assert "rtl/new.vhd" in second.untracked
    assert "doc/guide.md" not in second.untracked
    assert second.commit != first.commit

    third = run(manager.sync(cfg, second.commit))
    assert third.empty
    assert third.commit == second.commit


def test_filesystem_content_edit_moves_fingerprint(manager: GitManager, fs_root: Path):
    cfg = fs_config(fs_root)
    first = run(manager.sync(cfg, None))
    (fs_root / "rtl" / "a.vhd").write_text("entity a is end;\n-- v2\n")
    second = run(manager.sync(cfg, first.commit))
    # Same file set (the pipeline refines this to a re-chunk of a.vhd via
    # content hashes), but the walk fingerprint moves so the fast local
    # poller re-runs the sync.
    assert second.commit != first.commit
    assert sorted(second.untracked) == sorted(first.untracked)
    assert second.empty


def test_filesystem_local_fingerprint(manager: GitManager, fs_root: Path):
    cfg = fs_config(fs_root)
    fingerprint = run(manager.local_fingerprint(cfg))
    plan = run(manager.sync(cfg, None))
    assert fingerprint == plan.fingerprint
    (fs_root / "rtl" / "x.vhd").write_text("entity x is end;\n")
    assert run(manager.local_fingerprint(cfg)) != fingerprint


def test_filesystem_fingerprint_format(manager: GitManager, fs_root: Path):
    """Locks down the walk fingerprint's exact wire format: sha256 of
    ``"{path}\\t{mtime_ns}\\t{size}\\n"`` lines, sorted by path. The
    ``os.scandir``-based walk must keep producing this so existing
    indexes are not invalidated by the rewrite (``commit`` and
    ``local_fingerprint`` are the fingerprint itself)."""
    fingerprint, paths = manager._filesystem_fingerprint(fs_root)
    assert list(paths) == sorted(paths)
    expected_lines = []
    for rel in paths:
        st = (fs_root / rel).stat()
        expected_lines.append(f"{rel}\t{st.st_mtime_ns}\t{st.st_size}")
    expected_lines.sort()
    digest = hashlib.sha256()
    for line in expected_lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    assert fingerprint == digest.hexdigest()


def test_filesystem_fingerprint_skips_symlinked_directory(
    manager: GitManager, tmp_path: Path
):
    """A symlinked directory is neither descended into nor recorded
    itself, matching ``os.walk(..., followlinks=False)``."""
    root = tmp_path / "fsrepo2"
    real = tmp_path / "elsewhere"
    (real).mkdir(parents=True)
    (real / "outside.vhd").write_text("entity outside is end;\n")
    root.mkdir()
    (root / "rtl").mkdir()
    (root / "rtl" / "a.vhd").write_text("entity a is end;\n")
    (root / "link_dir").symlink_to(real, target_is_directory=True)
    _fingerprint, paths = manager._filesystem_fingerprint(root)
    assert paths == ("rtl/a.vhd",)


def test_filesystem_missing_path(manager: GitManager, tmp_path: Path):
    cfg = RepositoryConfig(name="missing", path=tmp_path / "nope", filesystem=True)
    with pytest.raises(GitError, match="does not exist"):
        run(manager.sync(cfg, None))
    with pytest.raises(GitError, match="does not exist"):
        run(manager.local_fingerprint(cfg))


# -- local repositories: index_untracked flag -----------------------------------


def test_index_untracked_flag_skips_untracked(manager: GitManager, tmp_path: Path):
    # A local working repository with one committed file and one
    # untracked file. (Remote/url repositories are cloned clean, so the
    # flag only has an effect for path repositories.)
    local = tmp_path / "workrepo"
    (local / "rtl").mkdir(parents=True)
    (local / "rtl" / "fifo.vhd").write_text("entity fifo is end;\n")
    git(local, "init", "-q", "-b", "main")
    git(local, "add", "-A")
    git(local, "commit", "-qm", "first")
    (local / "rtl" / "extra.vhd").write_text("entity extra is end;\n")

    cfg_off = RepositoryConfig(name="lrepo", path=local, index_untracked=False)
    first = run(manager.sync(cfg_off, None))
    assert "rtl/extra.vhd" not in first.added_or_modified
    assert first.untracked == ()

    # Flag off again: the untracked file stays out of the plan.
    second = run(manager.sync(cfg_off, first.commit))
    assert second.untracked == ()

    # Flag on: the same tree now reports the untracked file.
    cfg_on = RepositoryConfig(name="lrepo", path=local, index_untracked=True)
    third = run(manager.sync(cfg_on, first.commit))
    assert third.untracked == ("rtl/extra.vhd",)


# -- submodules ------------------------------------------------------------------


def test_submodule_full_sync(manager: GitManager, remote_with_submodule):
    up, _sub = remote_with_submodule
    cfg = make_config(up)
    plan = run(manager.sync(cfg, None))
    assert plan.full
    assert "top.vhd" in plan.added_or_modified
    # Submodule files are indexed under the gitlink path as prefix.
    assert "ip/rtl/a.vhd" in plan.added_or_modified
    assert plan.submodules and set(plan.submodules) == {"ip"}
    assert len(plan.submodules["ip"]) == 40


def bump_submodule(up: Path, sub_up: Path) -> None:
    """Point the superproject's gitlink at sub_up's current HEAD."""
    new_sha = git_live(sub_up, "rev-parse", "HEAD")
    git_live(up / "ip", "fetch", "origin")
    git_live(up / "ip", "checkout", "-q", new_sha)
    git_live(up, "add", "ip")
    git_live(up, "commit", "-qm", "bump submodule")


def test_submodule_pointer_change_indexes_new_files(
    manager: GitManager, remote_with_submodule
):
    up, sub_up = remote_with_submodule
    cfg = make_config(up)
    first = run(manager.sync(cfg, None))

    (sub_up / "rtl" / "b.vhd").write_text("entity b is end;\n")
    git_live(sub_up, "add", "-A")
    git_live(sub_up, "commit", "-qm", "sub second")
    bump_submodule(up, sub_up)

    second = run(manager.sync(cfg, first.commit, first.submodules))
    assert "ip/rtl/b.vhd" in second.added_or_modified
    # The unchanged submodule file is re-chunked too (whole-submodule
    # re-chunk on pointer change).
    assert "ip/rtl/a.vhd" in second.added_or_modified
    # Top-level files are untouched.
    assert "top.vhd" not in second.added_or_modified
    assert second.deleted == ()
    assert second.submodules["ip"] != first.submodules["ip"]


def test_submodule_file_deleted(manager: GitManager, remote_with_submodule):
    up, sub_up = remote_with_submodule
    cfg = make_config(up)
    first = run(manager.sync(cfg, None))

    (sub_up / "doc.txt").write_text("x\n")
    git_live(sub_up, "add", "-A")
    git_live(sub_up, "commit", "-qm", "add doc")
    bump_submodule(up, sub_up)
    second = run(manager.sync(cfg, first.commit, first.submodules))
    assert "ip/doc.txt" in second.added_or_modified

    git_live(sub_up, "rm", "-q", "doc.txt")
    git_live(sub_up, "commit", "-qm", "remove doc")
    bump_submodule(up, sub_up)
    third = run(manager.sync(cfg, second.commit, second.submodules))
    assert "ip/doc.txt" in third.deleted


def test_submodule_removed_purges_prefix(manager: GitManager, remote_with_submodule):
    up, _sub = remote_with_submodule
    cfg = make_config(up)
    first = run(manager.sync(cfg, None))

    git_live(up, "rm", "-q", "ip")
    git_live(up, "commit", "-qm", "remove submodule")
    second = run(manager.sync(cfg, first.commit, first.submodules))
    assert second.deleted_submodule_prefixes == ("ip",)
    assert second.submodules == {}
    assert not second.empty


def test_local_submodule_changes(
    manager: GitManager, tmp_path: Path, remote_with_submodule, sub_env: None
):
    up, _sub = remote_with_submodule
    local = tmp_path / "checkout"
    git_live(up, "clone", "-q", str(up), str(local))
    git_live(local, "submodule", "update", "--init")
    cfg = RepositoryConfig(name="repo", path=local)

    first = run(manager.sync(cfg, None))
    assert "ip/rtl/a.vhd" in first.added_or_modified
    assert first.submodules and set(first.submodules) == {"ip"}

    # Tracked file edited inside the submodule (uncommitted).
    (local / "ip" / "rtl" / "a.vhd").write_text("entity a is end;\n-- v2\n")
    second = run(manager.sync(cfg, first.commit, first.submodules))
    assert "ip/rtl/a.vhd" in second.added_or_modified
    assert "top.vhd" not in second.added_or_modified

    # Untracked file inside the submodule.
    (local / "ip" / "rtl" / "new.vhd").write_text("entity new is end;\n")
    third = run(manager.sync(cfg, first.commit, first.submodules))
    assert "ip/rtl/new.vhd" in third.untracked
    assert "ip/rtl/a.vhd" in third.added_or_modified

    # Submodule HEAD moves away from the recorded pointer: the whole
    # submodule is re-chunked (tracked + untracked files).
    git_live(local / "ip", "commit", "-qam", "sub local")
    fourth = run(manager.sync(cfg, first.commit, first.submodules))
    assert "ip/rtl/a.vhd" in fourth.added_or_modified
    assert "ip/rtl/new.vhd" in fourth.added_or_modified


def make_upstream_with_nested_submodule(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A superproject -> submodule ``ip`` -> nested submodule
    ``ip/vendor``; returns (up, sub_up, vendor_up)."""
    vendor_up = tmp_path / "vendor-upstream"
    vendor_up.mkdir()
    git_live(vendor_up, "init", "-q", "-b", "main")
    (vendor_up / "vendor.vhd").write_text("entity vendor is end;\n")
    git_live(vendor_up, "add", "-A")
    git_live(vendor_up, "commit", "-qm", "vendor first")

    sub_up = tmp_path / "sub-upstream"
    sub_up.mkdir()
    git_live(sub_up, "init", "-q", "-b", "main")
    (sub_up / "rtl").mkdir()
    (sub_up / "rtl" / "a.vhd").write_text("entity a is end;\n")
    git_live(sub_up, "add", "-A")
    git_live(sub_up, "commit", "-qm", "sub first")
    git_live(sub_up, "submodule", "add", "-q", str(vendor_up), "vendor")
    git_live(sub_up, "commit", "-qm", "add vendor submodule")

    up = tmp_path / "upstream"
    up.mkdir()
    git_live(up, "init", "-q", "-b", "main")
    (up / "top.vhd").write_text("entity top is end;\n")
    git_live(up, "add", "-A")
    git_live(up, "commit", "-qm", "super first")
    git_live(up, "submodule", "add", "-q", str(sub_up), "ip")
    git_live(up, "commit", "-qm", "add submodule")
    return up, sub_up, vendor_up


def test_local_nested_submodule_files_prefixed_once(
    manager: GitManager, tmp_path: Path, sub_env: None
):
    """Regression test for a submodule that itself has a submodule (e.g.
    a vendored VUnit checkout that pulls in OSVVM as its own submodule):
    the nested submodule's files must be prefixed with the full path
    from the repository root exactly once, not once per recursion level.
    """
    up, _sub, _vendor = make_upstream_with_nested_submodule(tmp_path)
    local = tmp_path / "checkout"
    git_live(up, "clone", "-q", str(up), str(local))
    git_live(local, "submodule", "update", "--init", "--recursive")
    cfg = RepositoryConfig(name="repo", path=local)

    plan = run(manager.sync(cfg, None))
    assert "top.vhd" in plan.added_or_modified
    assert "ip/rtl/a.vhd" in plan.added_or_modified
    assert "ip/vendor/vendor.vhd" in plan.added_or_modified
    # The double-prefix bug produced "ip/ip/vendor/vendor.vhd" instead.
    assert not any(p.startswith("ip/ip/") for p in plan.added_or_modified)
