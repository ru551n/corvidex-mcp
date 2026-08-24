"""Git repository management: clone, fetch, incremental change detection.

The manager owns the local clones (under ``<data_dir>/repos/<name>``) and
computes, for one configured repository, what changed between the last
indexed commit and the commit the configured ``ref`` resolves to:
added/modified files, renamed files, and deleted files. Nothing here
touches Qdrant or embeddings — the indexing pipeline consumes the
:class:`SyncPlan`.

Refs
----
``ref`` is any resolvable Git ref. A branch is tracked: every sync fetches
and moves with the remote branch. A tag or commit SHA pins the repository
to a fixed version; a full-SHA pin is resolved purely locally and never
touches the network.

All commands run with ``GIT_TERMINAL_PROMPT=0`` so an unattended server
can never hang on a credential prompt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import RepositoryConfig

logger = logging.getLogger(__name__)

CLONE_TIMEOUT = 600.0
FETCH_TIMEOUT = 300.0
GIT_TIMEOUT = 120.0


class GitError(RuntimeError):
    """A git operation failed."""


@dataclass(frozen=True)
class SyncPlan:
    """What the indexing pipeline must do for one repository.

    For a full plan, ``added_or_modified`` lists every file at the target
    commit and the pipeline deletes the repository from the store first.
    For an incremental plan it lists only the changed files (renames
    appear under their new path, with the old path in ``deleted``).
    """

    name: str
    ref: str
    commit: str
    full: bool
    added_or_modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """True when a sync found nothing to do (ref did not move)."""
        return not self.full and not self.added_or_modified and not self.deleted


def parse_name_status_z(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse ``git diff -z --name-status`` output.

    The ``-z`` form is a flat NUL-separated token stream: every entry is
    its status code followed by one path (``A``, ``M``, ``T``, ``D``) or
    two paths (``R100``, ``C100``: old then new).

    Returns ``(added_or_modified, deleted)``; a rename contributes its
    old path to ``deleted`` and its new path to ``added_or_modified``.
    """
    added: list[str] = []
    deleted: list[str] = []
    tokens = text.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue
        code = status[0]
        if code in ("R", "C"):
            if i + 2 >= len(tokens):
                break
            deleted.append(tokens[i + 1])
            added.append(tokens[i + 2])
            i += 3
        else:
            if i + 1 >= len(tokens):
                break
            path = tokens[i + 1]
            if code in ("A", "M", "T"):
                added.append(path)
            elif code == "D":
                deleted.append(path)
            i += 2
    return tuple(added), tuple(deleted)


class GitManager:
    """Clones and synchronizes repositories; reports changes as SyncPlans."""

    def __init__(self, repos_dir: Path) -> None:
        self._repos_dir = repos_dir

    def repo_dir(self, name: str) -> Path:
        return self._repos_dir / name

    # -- low-level git ----------------------------------------------------

    async def _git(
        self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from None
        return (
            proc.returncode if proc.returncode is not None else -1,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def _run(self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT) -> str:
        code, out, err = await self._git(cwd, *args, timeout=timeout)
        if code != 0:
            raise GitError(
                f"git {args[0]} failed (exit {code}): "
                f"{err.strip()[-500:] or out.strip()[-500:]}"
            )
        return out

    async def _try(
        self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT
    ) -> str | None:
        code, out, _ = await self._git(cwd, *args, timeout=timeout)
        return out if code == 0 else None

    # -- repository lifecycle ---------------------------------------------

    async def ensure_clone(self, cfg: RepositoryConfig) -> None:
        """Clone the repository on first use (idempotent)."""
        repo_dir = self.repo_dir(cfg.name)
        if (repo_dir / ".git").exists():
            return
        if repo_dir.exists():
            # Leftover of an interrupted clone: unusable without .git.
            logger.warning("removing incomplete clone directory %s", repo_dir)
            shutil.rmtree(repo_dir, ignore_errors=True)
        self._repos_dir.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s -> %s", cfg.url, repo_dir)
        await self._run(
            self._repos_dir,
            "clone",
            "--",
            cfg.url,
            str(repo_dir),
            timeout=CLONE_TIMEOUT,
        )

    async def _resolve(self, cfg: RepositoryConfig, repo_dir: Path) -> str:
        """Resolve the configured ref to a commit SHA (fetching if allowed)."""
        ref = cfg.ref
        if cfg.is_pinned_sha:
            commit = await self._try(
                repo_dir, "rev-parse", "--verify", f"{ref}^{{commit}}"
            )
            if commit is None:
                raise GitError(
                    f"pinned SHA {ref[:12]} not present in the local clone of "
                    f"{cfg.name!r}; full-SHA pins are never fetched, so the "
                    "commit must be reachable from a cloned branch"
                )
            return commit.strip()
        await self._run(
            repo_dir,
            "fetch",
            "--prune",
            "--tags",
            "origin",
            timeout=FETCH_TIMEOUT,
        )
        # Remote-tracking ref first: we check out detached HEADs, so a local
        # branch (if any) may still point at the previous commit. Tags and
        # plain SHAs only resolve as the ref itself.
        for candidate in (f"origin/{ref}", ref):
            commit = await self._try(
                repo_dir, "rev-parse", "--verify", f"{candidate}^{{commit}}"
            )
            if commit is not None:
                return commit.strip()
        raise GitError(f"ref {ref!r} does not resolve to a commit in {cfg.name!r}")

    async def _list_files_at(
        self, cfg: RepositoryConfig, commit: str
    ) -> tuple[str, ...]:
        repo_dir = self.repo_dir(cfg.name)
        out = await self._run(repo_dir, "ls-tree", "-r", "-z", "--name-only", commit)
        return tuple(path for path in out.split("\0") if path)

    async def sync(self, cfg: RepositoryConfig, last_commit: str | None) -> SyncPlan:
        """Synchronize the clone and report the changes vs ``last_commit``.

        ``last_commit`` is the last fully indexed commit (``None`` before
        the first index run). The working tree is left at the target
        commit.
        """
        await self.ensure_clone(cfg)
        repo_dir = self.repo_dir(cfg.name)
        commit = await self._resolve(cfg, repo_dir)
        head = await self._try(repo_dir, "rev-parse", "--verify", "HEAD")
        if head is None or head.strip() != commit:
            await self._run(repo_dir, "checkout", "--detach", commit)
        if last_commit is None:
            files = await self._list_files_at(cfg, commit)
            logger.info(
                "%s: first sync, full index of %d files at %s",
                cfg.name,
                len(files),
                commit[:12],
            )
            return SyncPlan(
                cfg.name, cfg.ref, commit, full=True, added_or_modified=files
            )
        if last_commit == commit:
            return SyncPlan(cfg.name, cfg.ref, commit, full=False)
        diff = await self._try(
            repo_dir, "diff", "-z", "--name-status", last_commit, commit
        )
        if diff is None:
            # Previous commit is gone (history rewrite / force push):
            # the safe fallback is a full reindex.
            logger.warning(
                "%s: cannot diff %s..%s; falling back to full reindex",
                cfg.name,
                last_commit[:12],
                commit[:12],
            )
            files = await self._list_files_at(cfg, commit)
            return SyncPlan(
                cfg.name, cfg.ref, commit, full=True, added_or_modified=files
            )
        added, deleted = parse_name_status_z(diff)
        logger.info(
            "%s: %s..%s: %d added/modified, %d deleted",
            cfg.name,
            last_commit[:12],
            commit[:12],
            len(added),
            len(deleted),
        )
        return SyncPlan(
            cfg.name,
            cfg.ref,
            commit,
            full=False,
            added_or_modified=added,
            deleted=deleted,
        )

    # -- file access --------------------------------------------------------

    def read_file(self, cfg: RepositoryConfig, relpath: str) -> str:
        """Read a file from the working tree (the tree sits at the last
        synced commit)."""
        if ".." in Path(relpath).parts or relpath.startswith("/"):
            raise GitError(f"invalid repository file path: {relpath!r}")
        path = self.repo_dir(cfg.name) / relpath
        if not path.is_file():
            raise GitError(f"file {relpath!r} not found in {cfg.name!r}")
        return path.read_text(encoding="utf-8", errors="replace")
