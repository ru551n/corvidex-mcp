"""Bundle embedding-model assets into the source tree.

Run before building the wheel: it provisions a fastembed TextEmbedding
model (default: the built-in default,
jinaai/jina-embeddings-v2-small-en) into
``src/corvidex_mcp/assets/<org>-<model>/`` with the exact layout
fastembed loads:

    <dest>/onnx/model.onnx
    <dest>/config.json
    <dest>/tokenizer.json
    <dest>/tokenizer_config.json
    <dest>/special_tokens_map.json

The ONNX weights are gitignored (they exceed GitHub's 100 MB per-file
limit) and the small JSON files are committed, so the wheel built from a
provisioned tree ships the model and an air-gapped install needs no
download at runtime.

Two provisioning paths:

* online — download through fastembed's own registry and HF cache:

      uv run --no-sync python tools/bundle_model.py
      uv run --no-sync python tools/bundle_model.py \
          --model jinaai/jina-embeddings-v2-small-en \
          --cache-dir ~/.cache/fastembed

* offline / sneakernet — copy from a directory that already holds the
  model files in the layout above (e.g. an HF cache snapshot):

      uv run --no-sync python tools/bundle_model.py --from /path/to/snapshot
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from corvidex_mcp.embeddings.assets import (
    ASSETS_DIR,
    MODEL_DIR_FILES,
    asset_subdir,
    bundled_model_dir,
)

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-small-en"


def resolve_description(model_name: str) -> tuple[Any, Any]:
    """(owner class, model description) from fastembed's registry."""
    from fastembed import TextEmbedding

    for cls in TextEmbedding.EMBEDDINGS_REGISTRY:
        if any(entry.model == model_name for entry in cls._list_supported_models()):
            return cls, cls._get_model_description(model_name)
    raise SystemExit(
        f"model {model_name!r} is not in the fastembed TextEmbedding registry"
    )


def copy_model(src_dir: Path, dest_dir: Path, model_file: str) -> None:
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
    print(f"copied {len(files)} files ({total / 1e6:.1f} MB) -> {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"fastembed TextEmbedding model name (default: {DEFAULT_MODEL})",
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
        help="copy from a directory holding the model files instead of downloading",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help=(
            "destination directory (default: "
            f"{ASSETS_DIR / asset_subdir(DEFAULT_MODEL)})"
        ),
    )
    args = parser.parse_args()

    cls, desc = resolve_description(args.model)
    dest = (
        Path(args.dest).expanduser()
        if args.dest
        else ASSETS_DIR / asset_subdir(args.model)
    )
    if args.from_dir is not None:
        src = Path(args.from_dir).expanduser().resolve()
        copy_model(src, dest, desc.model_file)
    else:
        cache = (
            Path(args.cache_dir).expanduser()
            if args.cache_dir
            else Path.home() / ".cache" / "corvidex-models"
        )
        cache.mkdir(parents=True, exist_ok=True)
        print(
            f"downloading {args.model} (registry source: {desc.sources.hf}) -> {cache}"
        )
        src = Path(cls.download_model(desc, str(cache)))
        copy_model(src, dest, desc.model_file)

    canonical = ASSETS_DIR / asset_subdir(args.model)
    if dest.resolve() == canonical.resolve():
        resolved = bundled_model_dir(args.model)
        if resolved is None or resolved.resolve() != dest.resolve():
            raise SystemExit(
                f"bundled_model_dir({args.model!r}) = {resolved!r}, expected {dest}; "
                "the files were not provisioned where the runtime looks"
            )
        print(f"ok: bundled_model_dir({args.model!r}) -> {resolved}")
        print(
            "the wheel built from this tree now ships the model: air-gapped "
            "installs need no download at runtime."
        )
    else:
        print(
            f"note: wrote to custom dest {dest}, which the runtime does not "
            f"load (it looks in {canonical}). Copy the files there (or rerun "
            "without --dest) to bundle the model."
        )


if __name__ == "__main__":
    main()
