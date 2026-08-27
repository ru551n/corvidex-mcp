"""Retrieval service: hybrid search, cross-domain fusion, source access.

Search is the vector store's native hybrid query (dense vectors +
full-text over the chunk content, RRF fusion) with row filters.

Cross-domain search (``search_knowledge``) fuses the per-collection
rank lists with RRF again (one list per collection, rank-based, so
the three domains interleave fairly regardless of their sizes).

Cross-referencing: callers may pass ``symbols`` — identifiers the
results must reference (stored on every chunk's payload); matching is
an OR over the identifiers, ANDed with the repository/category
filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import AppConfig
from .embeddings.providers import EmbeddingProviders
from .git_manager import GitError, GitManager
from .models import Chunk, CollectionName, SearchResult
from .state import StateStore
from .vector_store import ALL_COLLECTIONS, VectorStore

logger = logging.getLogger(__name__)

#: RRF rank constant (Qdrant's own FusionQuery default).
RRF_K = 60

#: Languages indexed into the hdl collection (routing-derived).
HDL_LANGUAGES = ("vhdl", "verilog", "systemverilog")


class RetrievalError(Exception):
    """A user-facing retrieval failure (bad arguments, missing data)."""


@dataclass(frozen=True)
class RepositoryStatus:
    """Indexing/sync state of one configured repository."""

    name: str
    ref: str
    domains: tuple[str, ...]
    indexed_commit: str | None
    indexed_at: str | None
    last_sync_at: str | None
    last_sync_error: str | None
    file_count: int


class RetrievalService:
    """Search and source access over the configured repositories."""

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

    # -- validation -----------------------------------------------------------

    def _repository(self, name: str | None) -> None:
        """Validate a repository name against the configured set."""
        if name is None:
            return
        known = [cfg.name for cfg in self._config.repositories]
        if name not in known:
            raise RetrievalError(
                f"unknown repository {name!r}; configured: {', '.join(known)}"
            )

    @staticmethod
    def _check_query(query: str) -> str:
        query = query.strip()
        if not query:
            raise RetrievalError("query must not be empty")
        return query

    @staticmethod
    def _check_language_value(language: str | None) -> str | None:
        if language is None:
            return None
        language = language.strip()
        if not language:
            raise RetrievalError("language must not be empty")
        return language

    # -- store-level search ------------------------------------------------------

    def _search_collection(
        self,
        collection: CollectionName,
        query: str,
        limit: int,
        repository: str | None,
        symbols: tuple[str, ...] | None,
        language: str | None,
    ) -> list[tuple[float, Chunk]]:
        """Hybrid query of one collection: (fused score, chunk) pairs."""
        dense = self._providers.embed_query(collection, query)
        must: dict[str, str] = {}
        if repository is not None:
            must["repository"] = repository
        if language is not None:
            must["language"] = language
        should = {"symbols": symbols} if symbols else None
        scored = self._store.query(
            collection,
            dense,
            query,
            limit=limit,
            must=must or None,
            should=should,
        )
        return [(sc.score, sc.chunk) for sc in scored]

    # -- public search ------------------------------------------------------------

    def search(
        self,
        collection: CollectionName,
        query: str,
        limit: int = 8,
        repository: str | None = None,
        symbols: tuple[str, ...] | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search in one collection.

        ``symbols`` restricts results to chunks referencing any of the
        given identifiers (cross-referencing). ``language`` restricts
        results to one payload language (e.g. ``verilog`` within the
        hdl collection); for the hdl collection the value is validated
        against :data:`HDL_LANGUAGES`.
        """
        query = self._check_query(query)
        self._repository(repository)
        language = self._check_language_value(language)
        if (
            language is not None
            and collection is CollectionName.HDL
            and language not in HDL_LANGUAGES
        ):
            raise RetrievalError(
                f"unknown HDL language {language!r}; expected one of: "
                + ", ".join(HDL_LANGUAGES)
            )
        pairs = self._search_collection(
            collection, query, limit, repository, symbols, language
        )
        return [self._to_result(score, chunk) for score, chunk in pairs]

    def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        repository: str | None = None,
        symbols: tuple[str, ...] | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search over all three collections, RRF-fused.

        Each collection contributes a rank list; the fused score is the
        sum of ``1/(RRF_K + rank)`` over the lists. ``language`` filters
        every collection (collections without that language simply
        contribute nothing).
        """
        query = self._check_query(query)
        self._repository(repository)
        language = self._check_language_value(language)
        fused: dict[tuple[str, str, int], tuple[float, Chunk]] = {}
        for collection in ALL_COLLECTIONS:
            pairs = self._search_collection(
                collection, query, limit, repository, symbols, language
            )
            # Rank, not the per-list score, is what fusion uses.
            for rank, (_score, chunk) in enumerate(pairs, start=1):
                key = (chunk.repository, chunk.file, chunk.start_line)
                entry = fused.get(key)
                term = 1.0 / (RRF_K + rank)
                if entry is None:
                    fused[key] = (term, chunk)
                else:
                    fused[key] = (entry[0] + term, entry[1])
        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item[0],
                item[1].repository,
                item[1].file,
                item[1].start_line,
            ),
        )
        return [self._to_result(score, chunk) for score, chunk in ranked[:limit]]

    # -- source access ------------------------------------------------------------

    def get_source(
        self,
        repository: str,
        file: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Full (or sliced) source of an indexed file, with attribution."""
        self._repository(repository)
        cfg = self._config.repository(repository)
        state = self._states.get(repository)
        if state.indexed_commit is None:
            raise RetrievalError(
                f"repository {repository!r} has no indexed commit yet; "
                "call sync_repositories first"
            )
        try:
            text = self._git.read_file(cfg, file)
        except GitError as exc:
            raise RetrievalError(str(exc)) from exc
        lines = text.splitlines()
        if not lines:
            raise RetrievalError(f"file {file!r} is empty")
        start = max(1, start_line if start_line is not None else 1)
        end = min(len(lines), end_line if end_line is not None else len(lines))
        if start > end:
            raise RetrievalError(
                f"invalid line range {start}-{end} (file has {len(lines)} lines)"
            )
        body = "\n".join(lines[start - 1 : end])
        return (
            f"{repository}:{file} @ {state.indexed_commit[:12]} "
            f"(lines {start}-{end} of {len(lines)})\n"
            f"{body}"
        )

    # -- status ----------------------------------------------------------------------

    def repository_status(self) -> list[RepositoryStatus]:
        """Indexing state for every configured repository."""
        statuses: list[RepositoryStatus] = []
        for cfg in self._config.repositories:
            state = self._states.get(cfg.name)
            statuses.append(
                RepositoryStatus(
                    name=cfg.name,
                    ref=cfg.ref,
                    domains=tuple(d.value for d in cfg.domains),
                    indexed_commit=state.indexed_commit,
                    indexed_at=(
                        state.indexed_at.isoformat()
                        if state.indexed_at is not None
                        else None
                    ),
                    last_sync_at=(
                        state.last_sync_at.isoformat()
                        if state.last_sync_at is not None
                        else None
                    ),
                    last_sync_error=state.last_sync_error,
                    file_count=state.last_indexed_file_count,
                )
            )
        return statuses

    # -- result assembly ------------------------------------------------------------

    def _to_result(self, score: float, chunk: Chunk) -> SearchResult:
        return SearchResult(
            result_type=chunk.collection.value,
            repository=chunk.repository,
            commit=chunk.commit,
            file=chunk.file,
            content=chunk.content,
            score=score,
            language=chunk.language,
            symbol=chunk.symbol,
            symbol_kind=chunk.symbol_kind,
            native_symbol_kind=chunk.native_symbol_kind,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            library=chunk.library,
            entity=chunk.entity,
            architecture=chunk.architecture,
            module=chunk.module,
            heading=chunk.heading,
            section=chunk.section,
            symbols=chunk.symbols,
        )
