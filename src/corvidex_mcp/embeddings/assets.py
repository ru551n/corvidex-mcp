"""Bundled embedding-model assets (air-gapped installs).

The default dense model's ONNX weights and tokenizer files can ship
inside the installed package (``assets/``); when they are present the
providers load the model from the bundled directory instead of
downloading it, so a runtime without network access needs no model
provisioning.

The ``.onnx`` weights are not committed to the repository (they exceed
GitHub's 100 MB per-file limit): ``tools/bundle_model.py`` provisions
them into this directory on the internet-connected machine that builds
the wheel. The small tokenizer/config files are committed.
"""

from __future__ import annotations

from pathlib import Path

#: Package asset root: one subdirectory per bundled model, named
#: ``<org>-<model>`` (the fastembed model name with ``/`` replaced).
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

#: Non-weights files fastembed needs inside a model directory (in
#: addition to the ``onnx/model.onnx`` weights).
MODEL_DIR_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

#: Relative path of the weights inside a model directory (per the
#: fastembed registry entry for the bundled model).
MODEL_WEIGHTS = "onnx/model.onnx"


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
