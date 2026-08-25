"""The indexing pipeline: incremental repository synchronization.

One :meth:`IndexPipeline.sync_repository` call moves one configured
repository from its last indexed commit to the commit its ``ref``
currently resolves to:

1. git sync (clone/fetch/diff) via :mod:`vhdl_rag_mcp.git_manager`;
2. deletion of stale chunks (whole repository for a full plan, per-file
   for deleted/renamed-away files);
3. per-domain chunking of the changed files (VHDL via vhdl_ls, docs,
   general code) respecting the repository's ``domains`` and ``exclude``;
4. embedding (per-collection dense + shared sparse) and upsert into the
   vector store;
5. state update — only after the index update fully succeeded, so a
   failed run leaves the previous commit as the last indexed one and the
   next sync retries the same diff.

All failures are contained per repository: one broken repository records
its error in the state store and does not affect the others.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..config import AppConfig, RepositoryConfig
from ..embeddings.providers import EmbeddingProviders
from ..git_manager import GitManager, SyncPlan
from ..lsp import VhdlLsp, default_libraries_dir
from ..models import Chunk, CollectionName, ContentType, SparseVectorData
from ..routing import FileKind, classify_file
from ..state import StateStore
from ..vector_store import VectorStore
from .code import chunk_code_file
from .docs import chunk_doc_file
from .vhdl import chunk_vhdl_file

logger = logging.getLogger(__name__)

VHDL_KIND = FileKind(ContentType.SOURCE, CollectionName.VHDL, "vhdl")


class IndexPipeline:
    """Coordinates git, chunkers, embeddings, and the vector store."""

    def __init__(
        self,
        config: AppConfig,
        git: GitManager,
        store: VectorStore,
        providers: EmbeddingProviders,
        states: StateStore,
    ) -> None:
        self._config = config
        self._git = git
        self._store = store
        self._providers = providers
        self._states = states
        # One sync per repository at a time: concurrent syncs would race
        # on the same git working tree (checkout/read interleaving).
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[name] = lock
        return lock

    async def sync_repository(self, cfg: RepositoryConfig) -> None:
        """Synchronize one repository from its last indexed commit."""
        async with self._lock_for(cfg.name):
            await self._sync(cfg, self._states.get(cfg.name).indexed_commit)

    async def reindex_repository(self, cfg: RepositoryConfig) -> None:
        """Force a full reindex of one repository (ignores last commit)."""
        async with self._lock_for(cfg.name):
            await self._sync(cfg, None)

    async def _sync(self, cfg: RepositoryConfig, last_commit: str | None) -> None:
        try:
            plan = await self._git.sync(cfg, last_commit)
        except Exception as exc:
            logger.exception("%s: git sync failed: %s", cfg.name, exc)
            self._states.record_sync(cfg.name, str(exc))
            raise
        if plan.empty:
            if last_commit is not None and plan.commit != last_commit:
                # HEAD/ref moved without a content change (an amend or
                # a force-push of an identical tree): the chunks are
                # still current, only the attribution commit changed.
                # Advance the indexed commit — otherwise the next sync
                # cannot diff against the rewritten-away commit and
                # falls back to a full reindex.
                logger.info(
                    "%s: commit moved %s -> %s without content change",
                    cfg.name,
                    last_commit[:12],
                    plan.commit[:12],
                )
                self._states.set_indexed(
                    cfg.name,
                    plan.commit,
                    file_count=self._states.get(cfg.name).last_indexed_file_count,
                )
            else:
                logger.info(
                    "%s: ref %r did not move (still at %s)",
                    cfg.name,
                    plan.ref,
                    plan.commit[:12],
                )
            self._states.record_sync(cfg.name, None)
            return
        logger.info(
            "%s: %s plan, %d files at %s%s",
            cfg.name,
            "FULL" if plan.full else "incremental",
            len(plan.added_or_modified),
            plan.commit[:12],
            f" (from {last_commit[:12]})" if last_commit else "",
        )
        try:
            await self._apply_plan(cfg, plan)
        except Exception as exc:
            # Keep the previous indexed commit so the next sync retries
            # the same diff.
            logger.exception(
                "%s: index update failed (%s); previous state kept",
                cfg.name,
                exc,
            )
            self._states.record_sync(cfg.name, str(exc))
            raise
        self._states.set_indexed(
            cfg.name, plan.commit, file_count=len(plan.added_or_modified)
        )
        self._states.record_sync(cfg.name, None)

    # -- plan application ----------------------------------------------------

    async def _apply_plan(self, cfg: RepositoryConfig, plan: SyncPlan) -> None:
        if plan.full:
            self._store.delete_repository(cfg.name)
        for f in plan.deleted:
            self._store.delete_file(cfg.name, f)

        files_by_kind: dict[FileKind, list[str]] = {}
        for f in plan.added_or_modified:
            kind = classify_file(f, cfg.enabled_collections, cfg.exclude)
            if kind is None:
                continue
            files_by_kind.setdefault(kind, []).append(f)

        # Chunk IDs embed the line range, so a modified file's new chunks
        # get new IDs: remove the file's previous chunks before upserting
        # (full plans already dropped the whole repository above).
        if not plan.full:
            for files in files_by_kind.values():
                for f in files:
                    self._store.delete_file(cfg.name, f)

        chunks: list[Chunk] = []
        vhdl_files = files_by_kind.pop(VHDL_KIND, [])
        if vhdl_files:
            chunks.extend(await self._chunk_vhdl_files(cfg, plan, vhdl_files))
        for kind, files in files_by_kind.items():
            for f in files:
                content = self._git.read_file(cfg, f)
                if kind.collection is CollectionName.DOCS:
                    chunks.extend(
                        chunk_doc_file(
                            cfg, f, content, plan.commit, kind.language, branch=plan.ref
                        )
                    )
                else:
                    chunks.extend(
                        chunk_code_file(
                            cfg, f, content, plan.commit, kind.language, branch=plan.ref
                        )
                    )

        if chunks:
            self._upsert(cfg, chunks)
            logger.info("%s: indexed %d chunks", cfg.name, len(chunks))

    async def _chunk_vhdl_files(
        self, cfg: RepositoryConfig, plan: SyncPlan, files: list[str]
    ) -> list[Chunk]:
        """Chunk VHDL files with one LSP session for the whole plan.

        All files are opened first and the server is waited on once, so
        the quiet window is not paid per file. Files with syntax errors
        get the structural fallback (the LSP tree is partial there).
        """
        repo_dir = self._git.repo_dir(cfg)
        lsp = VhdlLsp(
            self._config.vhdl_ls_path,
            repo_dir,
            libraries_dir=default_libraries_dir(self._config.vhdl_ls_path),
            vhdl_ls_hook=cfg.vhdl_ls_hook,
        )
        chunks: list[Chunk] = []
        try:
            await lsp.start()
            for f in files:
                await lsp.open_document(repo_dir / f)
            await lsp.wait_until_quiet(timeout=max(20.0, 2.0 * len(files)))
            for f in files:
                path: Path = repo_dir / f
                content = self._git.read_file(cfg, f)
                if lsp.has_syntax_error(path):
                    logger.info(
                        "%s: %s has syntax errors; using structural fallback",
                        cfg.name,
                        f,
                    )
                    symbols = None
                else:
                    symbols = await lsp.document_symbols(path)
                chunks.extend(
                    chunk_vhdl_file(
                        cfg,
                        f,
                        content,
                        plan.commit,
                        lsp_symbols=symbols,
                        branch=plan.ref,
                    )
                )
        finally:
            await lsp.shutdown()
        return chunks

    # -- embedding + upsert ---------------------------------------------------

    def _upsert(self, cfg: RepositoryConfig, chunks: list[Chunk]) -> None:
        """Embed (dense per collection, sparse shared) and upsert."""
        texts = [c.content for c in chunks]
        sparse_all: list[SparseVectorData] = self._providers.embed_sparse_passages(
            texts
        )
        indexes_by_collection: dict[CollectionName, list[int]] = {}
        for i, chunk in enumerate(chunks):
            indexes_by_collection.setdefault(chunk.collection, []).append(i)
        for collection, indexes in indexes_by_collection.items():
            items = [chunks[i] for i in indexes]
            dense = self._providers.embed_passages(
                collection, [c.content for c in items]
            )
            sparse = [sparse_all[i] for i in indexes]
            self._store.upsert_chunks(items, dense, sparse)

    # -- bulk maintenance ------------------------------------------------------

    def delete_repository(self, name: str) -> None:
        """Remove a repository's chunks and state (config removal)."""
        deleted = self._store.delete_repository(name)
        self._states.remove(name)
        logger.info("%s: removed %d chunks from all collections", name, deleted)
