"""Bundle model assets into the source tree (air-gapped wheels).

Run before building the wheel: it provisions fastembed models into
``src/corvidex_mcp/assets/<org>-<model>/`` with the exact layout
fastembed loads (identical for dense models and the cross-encoder
reranker):

    <dest>/onnx/model.onnx
    <dest>/config.json
    <dest>/tokenizer.json
    <dest>/tokenizer_config.json
    <dest>/special_tokens_map.json

The ONNX weights are gitignored (they exceed GitHub's *and* PyPI's
100 MB per-file limit) and the small JSON files are committed, so the
wheel built from a provisioned tree ships the models and an air-gapped
install needs no download at runtime.

``--all`` provisions *every* model the configuration actually loads —
the ``hdl``/``docs``/``code`` dense models plus the reranker — read
from the live ``EmbeddingsConfig`` defaults (see
``corvidex_mcp.embeddings.assets.required_models``), so it cannot drift
when a default changes. Without ``--all`` a single ``--model`` is
provisioned, as before.

Provisioning paths:

* online — download through fastembed's own registry and HF cache:

      uv run --no-sync python tools/bundle_model.py --all
      uv run --no-sync python tools/bundle_model.py \
          --model jinaai/jina-embeddings-v2-small-en \
          --cache-dir ~/.cache/fastembed

* offline / sneakernet — copy from a directory that already holds the
  files. ``--from`` accepts either a single model directory (one
  ``--model``) or a Hugging Face cache root holding
  ``models--<org>--<name>/snapshots/<sha>/`` subdirectories, which is
  what an existing ``<data_dir>/embed-cache`` is:

      uv run --no-sync python tools/bundle_model.py --from /path/to/snapshot
      uv run --no-sync python tools/bundle_model.py --all \
          --from ~/.local/share/corvidex/embed-cache

  Models missing from a cache root are downloaded instead; pass
  ``--offline`` to make a missing model an error rather than a fetch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from corvidex_mcp.embeddings.assets import (
    ASSETS_DIR,
    MODEL_DIR_FILES,
    BundledModel,
    ModelKind,
    asset_subdir,
    bundled_model_names,
    missing_models,
    required_models,
)

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-small-en"


def resolve_description(model_name: str) -> tuple[Any, Any, ModelKind]:
    """(owner class, model description, kind) from fastembed's registries.

    Dense models and cross-encoders live in separate registries with
    separate owner classes; both expose the same
    ``download_model``/``_get_model_description`` surface, so the rest of
    this tool treats them alike once the kind is known.
    """
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    registries = (
        (TextEmbedding.EMBEDDINGS_REGISTRY, ModelKind.EMBEDDING),
        (TextCrossEncoder.CROSS_ENCODER_REGISTRY, ModelKind.RERANKER),
    )
    for registry, kind in registries:
        for cls in registry:
            entries = cls._list_supported_models()
            if any(entry.model.lower() == model_name.lower() for entry in entries):
                return cls, cls._get_model_description(model_name), kind
    raise SystemExit(
        f"model {model_name!r} is in neither the fastembed TextEmbedding nor "
        "the TextCrossEncoder registry"
    )


def is_cache_root(path: Path) -> bool:
    """True for a Hugging Face cache directory (``models--*`` children).

    Distinguishes ``--from <embed-cache>`` (many models) from
    ``--from <snapshot>`` (the files of one model), so both keep working
    off the same flag.
    """
    return path.is_dir() and any(path.glob("models--*"))


def snapshot_in_cache(cache_root: Path, hf_repo: str) -> Path | None:
    """Newest snapshot directory for ``hf_repo`` inside an HF cache root.

    The cache directory name is case-sensitive on disk but the registry's
    repo id is not consistently cased (``xenova/...`` vs ``Xenova/...``),
    so the match is case-insensitive.
    """
    wanted = f"models--{hf_repo.replace('/', '--')}".lower()
    for child in cache_root.iterdir():
        if not child.is_dir() or child.name.lower() != wanted:
            continue
        snapshots = sorted(
            (p for p in (child / "snapshots").glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        return snapshots[-1] if snapshots else None
    return None


def copy_model(src_dir: Path, dest_dir: Path, model_file: str) -> int:
    """Copy one model's files into ``dest_dir``; returns bytes written."""
    files = [model_file, *MODEL_DIR_FILES]
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel in files:
        src = src_dir / rel
        if not src.is_file():
            raise SystemExit(f"missing {rel} in {src_dir}")
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)  # follows symlinks (HF cache layout)
        total += dst.stat().st_size
    print(f"  copied {len(files)} files ({total / 1e6:.1f} MB) -> {dest_dir}")
    return total


def provision(
    model_name: str,
    dest: Path,
    from_dir: Path | None,
    cache_dir: Path,
    offline: bool,
) -> int:
    """Provision one model into ``dest``; returns bytes written."""
    cls, desc, _kind = resolve_description(model_name)
    src: Path | None = None
    if from_dir is not None:
        if not from_dir.is_dir():
            # A cold cache directory that does not exist yet (e.g. a CI
            # cache miss) is a "nothing here", not a malformed snapshot:
            # fall through to the download below.
            print(f"  note: {from_dir} does not exist")
        elif is_cache_root(from_dir):
            src = snapshot_in_cache(from_dir, desc.sources.hf)
            if src is None:
                print(f"  note: no snapshot for {desc.sources.hf!r} in {from_dir}")
        else:
            src = from_dir
    if src is None:
        if offline:
            raise SystemExit(
                f"{model_name}: not available under --from and --offline "
                "was given, so it will not be downloaded"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  downloading {model_name} "
            f"(registry source: {desc.sources.hf}) -> {cache_dir}"
        )
        src = Path(cls.download_model(desc, str(cache_dir)))
    else:
        print(f"  source: {src}")
    return copy_model(src, dest, desc.model_file)


def selected_models(args: argparse.Namespace) -> tuple[BundledModel, ...]:
    if args.all:
        return required_models()
    _cls, _desc, kind = resolve_description(args.model)
    return (BundledModel(name=args.model, kind=kind, used_by=("--model",)),)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"single fastembed model name (default: {DEFAULT_MODEL})",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "provision every model the configuration loads (the hdl/docs/"
            "code dense models plus the reranker), read from the "
            "EmbeddingsConfig defaults"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "fastembed/HF download cache directory (default: ~/.cache/corvidex-models)"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_dir",
        default=None,
        help=(
            "copy from a single model directory, or from a Hugging Face "
            "cache root (e.g. <data_dir>/embed-cache), instead of downloading"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download: a model missing from --from is an error",
    )
    parser.add_argument(
        "--assets-root",
        default=None,
        help=(
            "package assets directory to provision into (default: this "
            f"checkout's {ASSETS_DIR}). Used by tools/build_release.py to "
            "provision an out-of-tree build copy."
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="explicit destination directory (single --model only)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the models the configuration needs, and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models = selected_models(args)

    if args.list:
        for model in required_models():
            print(model.describe())
        print(
            f"bundled in {ASSETS_DIR}: {', '.join(bundled_model_names()) or '(none)'}"
        )
        return

    if args.dest is not None and args.all:
        raise SystemExit("--dest applies to a single --model, not --all")

    assets_root = (
        Path(args.assets_root).expanduser().resolve()
        if args.assets_root
        else ASSETS_DIR
    )
    from_dir = Path(args.from_dir).expanduser().resolve() if args.from_dir else None
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".cache" / "corvidex-models"
    )

    total = 0
    for model in models:
        print(f"{model.describe()}")
        dest = (
            Path(args.dest).expanduser()
            if args.dest
            else assets_root / asset_subdir(model.name)
        )
        total += provision(model.name, dest, from_dir, cache_dir, args.offline)

    print(
        f"provisioned {len(models)} model(s), {total / 1e6:.1f} MB into {assets_root}"
    )
    if args.dest is not None:
        print(
            f"note: wrote to a custom --dest, which the runtime does not load "
            f"(it looks in {assets_root}); rerun without --dest to bundle."
        )
        return
    missing = missing_models(models, assets_root)
    if missing:
        raise SystemExit(
            f"not loadable from {assets_root}: {', '.join(missing)} — the "
            "files were not provisioned where the runtime looks"
        )
    loadable = ", ".join(bundled_model_names(assets_root))
    print(f"ok: loadable from {assets_root}: {loadable}")
    print(
        "the wheel built from this tree now ships these models: "
        "air-gapped installs need no download at runtime."
    )


if __name__ == "__main__":
    main()
