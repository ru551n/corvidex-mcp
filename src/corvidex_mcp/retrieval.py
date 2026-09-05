"""Retrieval service: search, cross-domain fusion, source access.

Each search leg is selectable per call (``mode``): ``hybrid`` (the
default — the vector store's native RRF fusion of dense vectors and
full-text over the chunk content), ``semantic`` (embedding similarity
only), or ``lexical`` (full-text/BM25 only), always with row filters.

Cross-domain search (``search_knowledge``) fuses the per-collection
rank lists with RRF again (one list per collection, rank-based, so
the three domains interleave fairly regardless of their sizes).

Cross-referencing: callers may pass ``symbols`` — identifiers the
results must reference (stored on every chunk's payload); matching is
an OR over the identifiers, ANDed with the repository/category
filters.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

from .config import CODING_STANDARDS_REPO, AppConfig, ConfigError
from .embeddings.providers import EmbeddingProviders
from .git_manager import GitError, GitManager
from .models import Chunk, CollectionName, SearchResult
from .retrieval_lexicon import expand_query
from .standards import StandardsError, extract_standards_text
from .state import StateStore
from .vector_store import ALL_COLLECTIONS, VectorStore

logger = logging.getLogger(__name__)

#: RRF rank constant (the standard value; originally taken from
#: Qdrant's FusionQuery default).
RRF_K = 60

#: Scale of the bounded repository-priority bonus (see
#: :meth:`RetrievalService._priority_bonus`). One priority unit is
#: worth one RRF adjacent-rank step at the top of a list; the bonus
#: saturates at a quarter of the best single-list term, so the maximum
#: possible bonus swing (0.5/(K+1)) is smaller than that term itself —
#: a chunk ranked in more lists than another always outranks it,
#: regardless of repository priority.
_PRIORITY_BONUS_STEP = 1.0 / (RRF_K * (RRF_K + 1))
_PRIORITY_BONUS_CAP = 0.25 / (RRF_K + 1)

#: Languages indexed into the hdl collection (routing-derived).
HDL_LANGUAGES = ("vhdl", "verilog", "systemverilog")

#: Search strategies: hybrid (dense + full-text, RRF-fused; the
#: default), semantic (dense leg only), lexical (full-text leg only).
SEARCH_MODES = ("hybrid", "semantic", "lexical")


class RetrievalError(Exception):
    """A user-facing retrieval failure (bad arguments, missing data)."""


@dataclass(frozen=True)
class RepositoryStatus:
    """Indexing/sync state of one configured repository."""

    name: str
    ref: str
    domains: tuple[str, ...]
    priority: int
    indexed_commit: str | None
    indexed_at: str | None
    last_sync_at: str | None
    last_sync_error: str | None
    file_count: int
    filesystem: bool = False


def _slice_text(
    repository: str,
    file: str,
    text: str,
    commit: str,
    start_line: int | None,
    end_line: int | None,
) -> str:
    """The source header plus a (default full) line slice of ``text``."""
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
        f"{repository}:{file} @ {commit[:12]} "
        f"(lines {start}-{end} of {len(lines)})\n"
        f"{body}"
    )


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
        """Validate a repository name against the configured set (the
        coding-standards pseudo-repository counts when configured)."""
        if name is None:
            return
        known = self._config.configured_repository_names()
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

    @staticmethod
    def _check_mode(mode: str) -> str:
        """Validate a search mode (case-insensitive; see SEARCH_MODES)."""
        mode = (mode or "").strip().lower()
        if mode not in SEARCH_MODES:
            raise RetrievalError(
                f"unknown search mode {mode!r}; expected one of: "
                + ", ".join(SEARCH_MODES)
            )
        return mode

    def _embed_query(self, collection: CollectionName, query: str) -> list[float]:
        """Query embedding with a clear error on a degraded model."""
        try:
            return self._providers.embed_query(collection, query)
        except Exception as exc:
            raise RetrievalError(
                f"the embedding model for the {collection.value} collection "
                f"is unavailable ({exc}); provision the model and retry, "
                "or search with mode='lexical'"
            ) from exc

    def _priority_bonus(self, repository: str) -> float:
        """Bounded post-RRF score bonus for a repository's priority.

        Each unit of ``priority`` (default 1) adds one RRF adjacent-rank
        step; the result is saturated at :data:`_PRIORITY_BONUS_CAP` so
        the bonus reorders chunks within a relevance tier but can never
        promote a chunk across tiers (see the constant's docstring).
        Unconfigured repositories get no bonus. The coding-standards
        pseudo-repository uses ``coding_standards_priority``.
        """
        if repository == CODING_STANDARDS_REPO:
            if self._config.coding_standards is None:
                return 0.0
            priority = self._config.coding_standards_priority
        else:
            try:
                priority = self._config.repository(repository).priority
            except ConfigError:
                return 0.0
        delta = priority - 1
        if delta == 0:
            return 0.0
        bonus = delta * _PRIORITY_BONUS_STEP
        return max(-_PRIORITY_BONUS_CAP, min(_PRIORITY_BONUS_CAP, bonus))

    # -- store-level search ------------------------------------------------------

    def _search_collection(
        self,
        collection: CollectionName,
        query: str,
        limit: int,
        repository: str | None,
        symbols: tuple[str, ...] | None,
        language: str | None,
        mode: str,
    ) -> list[tuple[float, Chunk]]:
        """One collection in the given search mode: (score, chunk) pairs.

        The query is expanded with RTL/HDL domain synonyms before both
        legs when ``query_expansion_enabled`` (see
        :mod:`.retrieval_lexicon`). When ``rerank_enabled``, more than
        ``limit`` candidates are fetched from the store and a
        cross-encoder reranks them (on the original, unexpanded query)
        before truncating back to ``limit`` (see :meth:`_rerank`).
        """
        embeddings = self._config.embeddings
        expanded = expand_query(query) if embeddings.query_expansion_enabled else query
        fetch_limit = (
            max(limit, embeddings.rerank_candidates)
            if embeddings.rerank_enabled
            else limit
        )
        # The lexical leg never embeds the query: no model work at all.
        dense: list[float] = []
        if mode != "lexical":
            dense = self._embed_query(collection, expanded)
        must: dict[str, str] = {}
        if repository is not None:
            must["repository"] = repository
        if language is not None:
            must["language"] = language
        should = {"symbols": symbols} if symbols else None
        scored = self._store.query(
            collection,
            dense,
            expanded,
            limit=fetch_limit,
            must=must or None,
            should=should,
            mode=mode,
        )
        pairs = [(sc.score, sc.chunk) for sc in scored]
        return self._rerank(query, pairs, limit)

    def _rerank(
        self, query: str, pairs: list[tuple[float, Chunk]], limit: int
    ) -> list[tuple[float, Chunk]]:
        """Cross-encoder rerank of ``pairs``, truncated to ``limit``.

        Falls back to the input ranking (truncated to ``limit``) when
        reranking is disabled, there is nothing to rerank, or the
        model fails to load/run (not provisioned, no network yet) — a
        missing reranker degrades precision, it never fails the
        search. The returned score is the sigmoid-normalized
        cross-encoder relevance in ``(0, 1)`` (see
        :class:`corvidex_mcp.embeddings.reranker.CrossEncoderReranker`),
        the same scale family the RRF/cosine scores it replaces use, so
        it composes with the bounded per-repository priority bonus the
        same way.
        """
        if not self._config.embeddings.rerank_enabled or len(pairs) <= 1:
            return pairs[:limit]
        texts = [chunk.content for _, chunk in pairs]
        try:
            scores = self._providers.rerank(query, texts)
        except Exception as exc:
            logger.warning(
                "reranking unavailable (%s); returning the unreranked ranking",
                exc,
            )
            return pairs[:limit]
        reranked = sorted(
            zip(scores, (chunk for _, chunk in pairs), strict=True),
            key=lambda item: -item[0],
        )
        return reranked[:limit]

    # -- public search ------------------------------------------------------------

    def search(
        self,
        collection: CollectionName,
        query: str,
        limit: int = 8,
        repository: str | None = None,
        symbols: tuple[str, ...] | None = None,
        language: str | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Search one collection.

        ``mode`` selects the strategy: ``hybrid`` (default; dense +
        full-text, RRF-fused), ``semantic`` (dense only), ``lexical``
        (full-text only) — see :data:`SEARCH_MODES`. ``symbols``
        restricts results to chunks referencing any of the given
        identifiers (cross-referencing). ``language`` restricts results
        to one payload language (e.g. ``verilog`` within the hdl
        collection); for the hdl collection the value is validated
        against :data:`HDL_LANGUAGES`. The query is expanded and the
        candidates cross-encoder reranked before this returns (see
        :meth:`_search_collection`, :meth:`_rerank`). The score carries
        a bounded per-repository priority bonus (see
        :meth:`_priority_bonus`).
        """
        query = self._check_query(query)
        self._repository(repository)
        language = self._check_language_value(language)
        mode = self._check_mode(mode)
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
            collection, query, limit, repository, symbols, language, mode
        )
        boosted = [
            (score + self._priority_bonus(chunk.repository), chunk)
            for score, chunk in pairs
        ]
        boosted.sort(
            key=lambda item: (
                -item[0],
                item[1].repository,
                item[1].file,
                item[1].start_line,
            )
        )
        return [self._to_result(score, chunk) for score, chunk in boosted]

    def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        repository: str | None = None,
        symbols: tuple[str, ...] | None = None,
        language: str | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Search all three collections in one strategy, RRF-fused.

        Each collection is queried in the given ``mode`` (see
        :data:`SEARCH_MODES`) and contributes a rank list; the fused
        score is the sum of ``1/(RRF_K + rank)`` over the lists, plus
        the bounded per-repository priority bonus (see
        :meth:`_priority_bonus`). ``language`` filters every collection
        (collections without that language simply contribute nothing).
        """
        query = self._check_query(query)
        self._repository(repository)
        language = self._check_language_value(language)
        mode = self._check_mode(mode)
        fused: dict[tuple[str, str, int], tuple[float, Chunk]] = {}
        for collection in ALL_COLLECTIONS:
            if mode != "lexical":
                try:
                    self._providers.dimension(collection)  # load the model
                except Exception as exc:
                    logger.warning(
                        "search_knowledge: skipping the %s collection: "
                        "embedding model unavailable (%s)",
                        collection.value,
                        exc,
                    )
                    continue
            pairs = self._search_collection(
                collection, query, limit, repository, symbols, language, mode
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
                -(item[0] + self._priority_bonus(item[1].repository)),
                item[1].repository,
                item[1].file,
                item[1].start_line,
            ),
        )
        return [
            self._to_result(score + self._priority_bonus(chunk.repository), chunk)
            for score, chunk in ranked[:limit]
        ]

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
        if repository == CODING_STANDARDS_REPO:
            return self._standards_source(file, start_line, end_line)
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
        return _slice_text(
            repository, file, text, state.indexed_commit, start_line, end_line
        )

    def _standards_source(
        self,
        file: str,
        start_line: int | None,
        end_line: int | None,
    ) -> str:
        """Source of the coding-standards file (extracted text; line
        numbers refer to the indexed extraction)."""
        path = self._config.coding_standards
        assert path is not None  # validated by _repository
        if file != path.name:
            raise RetrievalError(
                f"the coding-standards repository holds a single file: {path.name!r}"
            )
        state = self._states.get(CODING_STANDARDS_REPO)
        if state.indexed_commit is None:
            raise RetrievalError(
                "the coding-standards file has not been indexed yet; "
                "call sync_repositories first"
            )
        try:
            text = extract_standards_text(path)
        except StandardsError as exc:
            raise RetrievalError(str(exc)) from exc
        return _slice_text(
            CODING_STANDARDS_REPO,
            path.name,
            text,
            state.indexed_commit,
            start_line,
            end_line,
        )

    def list_files(
        self,
        repository: str,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[str], bool]:
        """Indexed file paths of one repository (glob-filtered, capped).

        ``pattern`` is an fnmatch glob matched against the
        repository-relative path (``*`` crosses ``/``). Returns
        ``(paths, truncated)``; ``truncated`` is True when the cap hid
        more results than returned.
        """
        self._repository(repository)
        files = self._store.list_files(repository)
        if pattern:
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        truncated = limit is not None and len(files) > limit
        return (files[:limit] if limit is not None else files), truncated

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
                    priority=cfg.priority,
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
                    filesystem=cfg.filesystem,
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
