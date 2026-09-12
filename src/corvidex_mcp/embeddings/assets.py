"""Bundled model assets (air-gapped installs).

Every model the server loads — the per-collection dense embedding
models *and* the cross-encoder reranker — can ship inside the installed
package (``assets/``); when a model's files are present the providers
load it from the bundled directory instead of downloading it, so a
runtime without network access needs no model provisioning at all.

The ``.onnx`` weights are not committed to the repository (they exceed
GitHub's 100 MB per-file limit): ``tools/bundle_model.py`` provisions
them into this directory on the machine that builds the wheel. The
small tokenizer/config files are committed.

:func:`required_models` is the single inventory of what "all models"
means. It is derived from the live :class:`~corvidex_mcp.config
.EmbeddingsConfig` field values rather than from a hand-kept list, so
the bundling tool cannot drift when a default model changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..config import EmbeddingsConfig

#: Package asset root: one subdirectory per bundled model, named
#: ``<org>-<model>`` (the fastembed model name with ``/`` replaced).
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

#: Non-weights files fastembed needs inside a model directory (in
#: addition to the ``onnx/model.onnx`` weights). The dense models and
#: the cross-encoder reranker use the same layout.
MODEL_DIR_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

#: Relative path of the weights inside a model directory (per the
#: fastembed registry entry for the bundled model).
MODEL_WEIGHTS = "onnx/model.onnx"


class ModelKind(Enum):
    """Which fastembed loader a bundled model belongs to.

    The two kinds share a file layout but not a registry: dense models
    live in ``TextEmbedding.EMBEDDINGS_REGISTRY`` and the reranker in
    ``TextCrossEncoder.CROSS_ENCODER_REGISTRY``.
    """

    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass(frozen=True)
class BundledModel:
    """One model the configuration asks for, and what uses it."""

    #: fastembed model name, e.g. ``jinaai/jina-embeddings-v2-small-en``.
    name: str
    kind: ModelKind
    #: Human-readable consumers, e.g. ``("hdl", "docs")``.
    used_by: tuple[str, ...]

    @property
    def subdir(self) -> str:
        return asset_subdir(self.name)

    def describe(self) -> str:
        return f"{self.name} [{self.kind.value}] <- {', '.join(self.used_by)}"


def required_models(
    embeddings: EmbeddingsConfig | None = None,
) -> tuple[BundledModel, ...]:
    """Every model the given embeddings configuration loads, deduplicated.

    With ``None`` the built-in defaults are used. The field *values* are
    read directly, so changing a default in ``config.py`` automatically
    changes what ``tools/bundle_model.py --all`` provisions — there is no
    second list to keep in sync.

    ``hdl`` and ``docs`` share one model by default: it is returned once,
    with both consumers listed in :attr:`BundledModel.used_by`.
    """
    e = embeddings if embeddings is not None else EmbeddingsConfig()
    wanted: list[tuple[str, ModelKind, str]] = [
        (e.hdl_model, ModelKind.EMBEDDING, "hdl"),
        (e.docs_model, ModelKind.EMBEDDING, "docs"),
        (e.code_model, ModelKind.EMBEDDING, "code"),
        (e.rerank_model, ModelKind.RERANKER, "rerank"),
    ]
    merged: dict[str, tuple[ModelKind, list[str]]] = {}
    for name, kind, consumer in wanted:
        entry = merged.get(name)
        if entry is None:
            merged[name] = (kind, [consumer])
        else:
            entry[1].append(consumer)
    return tuple(
        BundledModel(name=name, kind=kind, used_by=tuple(consumers))
        for name, (kind, consumers) in merged.items()
    )


def asset_subdir(model_name: str) -> str:
    """Asset subdirectory name for a fastembed model name."""
    return model_name.replace("/", "-")


def bundled_model_dir(model_name: str) -> Path | None:
    """Directory of the model bundled in the package, if provisioned.

    Returns ``None`` when the model is not bundled or its files are
    missing, in which case the provider falls back to the fastembed
    download/cache behavior.
    """
    root = ASSETS_DIR / asset_subdir(model_name)
    if not (root / MODEL_WEIGHTS).is_file():
        return None
    for name in MODEL_DIR_FILES:
        if not (root / name).is_file():
            return None
    return root


def bundled_model_names(root: Path | None = None) -> tuple[str, ...]:
    """Asset subdirectory names that hold a complete, loadable model.

    Diagnostic only (the runtime looks models up by name): it answers
    "which models does this installation actually carry?" — used by the
    offline-bundle verification and by ``tools/bundle_model.py``, which
    passes an out-of-tree ``root`` when provisioning a build copy.
    """
    root = root if root is not None else ASSETS_DIR
    if not root.is_dir():
        return ()
    found: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / MODEL_WEIGHTS).is_file():
            continue
        if any(not (child / name).is_file() for name in MODEL_DIR_FILES):
            continue
        found.append(child.name)
    return tuple(found)


def missing_models(
    models: tuple[BundledModel, ...], root: Path | None = None
) -> tuple[str, ...]:
    """Names of ``models`` whose files are not (completely) provisioned."""
    present = set(bundled_model_names(root))
    return tuple(m.name for m in models if m.subdir not in present)
