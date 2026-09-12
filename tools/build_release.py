"""Build the two distribution artifacts: the PyPI wheel and the offline bundle.

They cannot be one artifact. PyPI enforces a 100 MB per-file limit (and
a per-project quota), and limit increases are not granted for bundled
model weights — the three models this server loads are ~0.86 GB of ONNX
weights. So:

* **slim** — ``dist/corvidex_mcp-<version>-py3-none-any.whl`` plus the
  sdist: no ``assets/*/onnx/`` weights, models fetched on first run
  (the current default behaviour). This is what ``twine upload`` gets.

* **offline** —
  ``dist-offline/corvidex-mcp-<version>-offline-<platform>-<abi>.tar.gz``:
  a wheel built *with* every model embedded, a dependency wheelhouse, an
  install script, and a README. Distributed outside PyPI (a GitHub
  Release asset), for air-gapped hosts. It lands in ``dist-offline/``,
  never ``dist/``, so ``twine check dist/*`` and an eventual
  ``twine upload dist/*`` can never see it.

Splitting the weights across several sub-100 MB wheels to squeeze them
onto PyPI was considered and rejected: it abuses a shared, donated
index to host ~0.8 GB of model binaries that are not this project's
code, and it is exactly what the limit exists to prevent.

The two wheels must never be confusable. The embedded one carries a
PEP 440 **local version segment**, ``<version>+offline``:

* PyPI rejects local versions outright (``twine upload`` fails before
  any byte of a 0.8 GB wheel is sent), so it cannot be published by
  accident;
* ``pip`` still installs it, and local versions compare *higher* than
  the public release, so an offline host that later sees an index does
  not get silently downgraded;
* ``pip show corvidex-mcp`` says ``0.1.0+offline`` — the user can tell
  which one they have without inspecting the wheel.

Both builds run in a throwaway copy of the checkout under ``--work-dir``
so the working tree is never mutated (in particular: the slim wheel
cannot accidentally pick up weights left behind by an earlier offline
build).

Usage::

    uv run --no-sync python tools/build_release.py --all \\
        --model-source ~/.local/share/corvidex/embed-cache
    uv run --no-sync python tools/build_release.py --slim
    uv run --no-sync python tools/build_release.py --offline \\
        --model-source ~/.local/share/corvidex/embed-cache
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Local version segment marking the all-models wheel (see module docstring).
OFFLINE_LOCAL_VERSION = "offline"

#: Wheel platform tags the wheelhouse is built for. pip matches
#: ``--platform`` values literally against wheel tags — it does not treat
#: manylinux_2_28 as satisfying a request for manylinux2014 — so every
#: variant a dependency may publish has to be listed, or that dependency
#: silently has no candidate at all. The *effective* floor of the
#: resulting wheelhouse is the strictest tag actually chosen, not the
#: loosest tag requested: today that is tree-sitter-language-pack's
#: manylinux_2_34 (glibc 2.34, i.e. RHEL 9 / Ubuntu 22.04 and newer).
DEFAULT_PLATFORMS = [
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_34_x86_64",
]

#: Paths never copied into a build tree (build output, caches, VCS).
COPY_EXCLUDES = shutil.ignore_patterns(
    ".git",
    ".venv",
    "dist",
    "build",
    "*.egg-info",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
)


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def project_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version: str = data["project"]["version"]
    return version


def human(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def fresh_copy(work_dir: Path, name: str) -> Path:
    """A clean copy of the checkout at ``work_dir/name``."""
    dest = work_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT, dest, ignore=COPY_EXCLUDES, symlinks=True)
    # A source checkout may already carry weights from an earlier offline
    # build; the slim build must never inherit them.
    for onnx_dir in (dest / "src" / "corvidex_mcp" / "assets").glob("*/onnx"):
        shutil.rmtree(onnx_dir)
    return dest


def set_local_version(tree: Path, local: str) -> str:
    """Append a PEP 440 local version segment to the build copy's version."""
    pyproject = tree / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    base = project_version(tree)
    marked = f"{base}+{local}"
    needle = f'version = "{base}"'
    if needle not in text:
        raise SystemExit(f"cannot find {needle!r} in {pyproject}")
    pyproject.write_text(text.replace(needle, f'version = "{marked}"', 1), "utf-8")
    return marked


def dist_files(out_dir: Path) -> set[Path]:
    """Distribution files in ``out_dir`` — uv also drops a ``.gitignore``
    there, which is neither an artifact nor something to ship."""
    if not out_dir.exists():
        return set()
    return {p for p in out_dir.glob("*") if p.suffix in (".whl", ".gz")}


def uv_build(tree: Path, out_dir: Path, wheel_only: bool) -> list[Path]:
    before = dist_files(out_dir)
    cmd = ["uv", "build", "--out-dir", str(out_dir)]
    if wheel_only:
        cmd.append("--wheel")
    run(cmd, cwd=tree)
    return sorted(dist_files(out_dir) - before)


def build_slim(work_dir: Path, dist: Path) -> list[Path]:
    print("== slim (PyPI) wheel + sdist ==")
    tree = fresh_copy(work_dir, "slim")
    built = uv_build(tree, dist, wheel_only=False)
    shutil.rmtree(tree)
    return built


def provision_models(tree: Path, model_source: Path | None, offline: bool) -> None:
    cmd = [
        sys.executable,
        str(tree / "tools" / "bundle_model.py"),
        "--all",
        "--assets-root",
        str(tree / "src" / "corvidex_mcp" / "assets"),
    ]
    if model_source is not None:
        cmd += ["--from", str(model_source)]
    if offline:
        cmd.append("--offline")
    env = {**os.environ, "PYTHONPATH": str(tree / "src")}
    run(cmd, cwd=tree, env=env)


def build_wheelhouse(
    tree: Path,
    dest: Path,
    python_version: str,
    platforms: list[str],
) -> None:
    """Download every runtime dependency as a wheel into ``dest``.

    Binary-only and pinned to the target interpreter/platform, so the
    air-gapped host never has to build anything (it has no index and
    quite possibly no compiler).
    """
    requirements = dest.parent.parent / "requirements-bundle.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requirements.open("w", encoding="utf-8") as fh:
        subprocess.run(
            ["uv", "export", "--no-hashes", "--no-emit-project", "--no-dev"],
            cwd=tree,
            check=True,
            stdout=fh,
        )
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "pip",
        "python",
        "-m",
        "pip",
        "download",
        "-r",
        str(requirements),
        "--dest",
        str(dest),
        "--only-binary=:all:",
        "--python-version",
        python_version,
    ]
    for platform in platforms:
        cmd += ["--platform", platform]
    run(cmd, cwd=tree)


INSTALL_SH = """\
#!/bin/sh
# Offline installer for corvidex-mcp. No network access is used or needed.
set -eu

target="${1:-$(pwd)/corvidex-venv}"
here=$(cd "$(dirname "$0")" && pwd)
python="${PYTHON:-python3.12}"

if ! command -v "$python" >/dev/null 2>&1; then
    echo "error: $python not found. Set PYTHON=/path/to/python3.12+ and retry." >&2
    exit 1
fi

echo "creating venv: $target"
"$python" -m venv "$target"

echo "installing from the bundled wheels (no index)"
"$target/bin/pip" install --no-index --find-links "$here/wheelhouse" \\
    --find-links "$here" "$here"/corvidex_mcp-*.whl

echo
version=$("$target/bin/pip" show corvidex-mcp | sed -n 's/^Version: //p')
echo "installed:  corvidex-mcp $version"
echo "server:    $target/bin/corvidex-mcp"
echo
echo "verify (still offline):"
echo "  $target/bin/python -m corvidex_mcp.verify_offline"
echo
echo "register with an MCP client, e.g.:"
echo "  claude mcp add corvidex-mcp -- $target/bin/corvidex-mcp"
"""


BUNDLE_README = """\
# corvidex-mcp {version} — offline bundle ({platform_label})

Everything needed to install and run `corvidex-mcp` on a host with **no
network access**: the package wheel with all three models embedded, a
wheel for every runtime dependency, and an installer.

This bundle is **not on PyPI** and never will be. PyPI enforces a 100 MB
per-file limit and does not grant increases for bundled model weights;
the embedded models are {weights_size} of ONNX weights, so the wheel in
here is {wheel_size}. PyPI carries the slim wheel
(`pip install corvidex-mcp`, models downloaded on first run); this
bundle is distributed as a GitHub Release asset instead.

## Contents

* `corvidex_mcp-{marked_version}-py3-none-any.whl` — the package with
  every model embedded ({wheel_size})
* `wheelhouse/` — every runtime dependency as a wheel
  ({wheelhouse_count} files, {wheelhouse_size})
* `install.sh` — creates a venv and installs both, with `--no-index`
* `README.md` — this file

The version is `{marked_version}`, not `{version}`: the `+offline`
suffix is a PEP 440 local version segment. It marks this build as the
all-models variant (`pip show corvidex-mcp` reports it), it compares
higher than the plain `{version}` on PyPI so an offline host is never
silently downgraded, and PyPI refuses local versions outright, so this
wheel cannot be uploaded there by mistake.

## Requirements on the target host

* Python {python_requires} ({python_label} for this bundle's wheels) with
  `venv` and `pip`
* `git` (repositories are indexed from Git working trees)
* `{platform_label}`
* `vhdl_ls` / Veridian are **optional** — without them HDL files fall
  back to structural parsing.

## Install

```console
$ tar xzf {archive_name}
$ cd {bundle_dir}
$ ./install.sh /opt/corvidex-venv
```

`PYTHON=/usr/bin/python3.13 ./install.sh ...` selects a different
interpreter (it must match the wheel tags in `wheelhouse/`).

## Verify (offline)

```console
$ /opt/corvidex-venv/bin/python -m corvidex_mcp.verify_offline
```

It reports which models are loaded from the bundled assets, indexes a
throwaway repository through the real pipeline, and runs a search over
all three collections with reranking on — with every outbound socket
refused, so a silent download would fail rather than hide. Expect a
final `offline verification: OK` line.

## Register with an MCP client

```console
$ claude mcp add corvidex-mcp -- /opt/corvidex-venv/bin/corvidex-mcp
```

The server indexes the directory it is started in; see
`docs/configuration.md` in the repository for `.corvidex` options.

## What is in the wheel

| Model | Role | Weights |
| --- | --- | --- |
{model_rows}

They live in `corvidex_mcp/assets/<org>-<model>/` inside the wheel and
are loaded from there directly (fastembed `specific_model_path`), so no
Hugging Face cache lookup and no download ever happens.
"""


def write_bundle_docs(
    bundle: Path,
    *,
    version: str,
    marked_version: str,
    wheel: Path,
    wheelhouse: Path,
    archive_name: str,
    python_label: str,
    platform_label: str,
    python_requires: str,
    models: list[tuple[str, str, int]],
) -> None:
    wheelhouse_files = sorted(wheelhouse.glob("*"))
    weights_total = sum(size for _, _, size in models)
    rows = "\n".join(
        f"| `{name}` | {role} | {human(size)} |" for name, role, size in models
    )
    (bundle / "README.md").write_text(
        BUNDLE_README.format(
            version=version,
            marked_version=marked_version,
            wheel_size=human(wheel.stat().st_size),
            weights_size=human(weights_total),
            wheelhouse_count=len(wheelhouse_files),
            wheelhouse_size=human(sum(p.stat().st_size for p in wheelhouse_files)),
            archive_name=archive_name,
            bundle_dir=bundle.name,
            python_label=python_label,
            platform_label=platform_label,
            python_requires=python_requires,
            model_rows=rows,
        ),
        encoding="utf-8",
    )
    install = bundle / "install.sh"
    install.write_text(INSTALL_SH, encoding="utf-8")
    install.chmod(0o755)


def build_offline(
    work_dir: Path,
    offline_dist: Path,
    model_source: Path | None,
    models_offline: bool,
    python_version: str,
    platforms: list[str],
    platform_label: str,
    abi_label: str,
) -> Path:
    print("== offline bundle (all models embedded) ==")
    tree = fresh_copy(work_dir, "offline")
    version = project_version(tree)
    marked = set_local_version(tree, OFFLINE_LOCAL_VERSION)
    provision_models(tree, model_source, models_offline)

    assets = tree / "src" / "corvidex_mcp" / "assets"
    model_meta: list[tuple[str, str, int]] = []
    sys.path.insert(0, str(tree / "src"))
    try:
        from corvidex_mcp.embeddings.assets import (
            MODEL_WEIGHTS,
            required_models,
        )

        for model in required_models():
            weights = assets / model.subdir / MODEL_WEIGHTS
            model_meta.append(
                (model.name, ", ".join(model.used_by), weights.stat().st_size)
            )
    finally:
        sys.path.pop(0)

    staging = work_dir / "bundle"
    if staging.exists():
        shutil.rmtree(staging)
    bundle = staging / f"corvidex-mcp-{version}-offline-{platform_label}-{abi_label}"
    bundle.mkdir(parents=True)

    (wheel,) = uv_build(tree, bundle, wheel_only=True)
    (bundle / ".gitignore").unlink(missing_ok=True)  # uv's, not ours
    build_wheelhouse(tree, bundle / "wheelhouse", python_version, platforms)

    archive_name = f"{bundle.name}.tar.gz"
    write_bundle_docs(
        bundle,
        version=version,
        marked_version=marked,
        wheel=wheel,
        wheelhouse=bundle / "wheelhouse",
        archive_name=archive_name,
        python_label=f"CPython {python_version}",
        platform_label=platform_label,
        python_requires=">= 3.12",
        models=model_meta,
    )

    offline_dist.mkdir(parents=True, exist_ok=True)
    archive = offline_dist / archive_name
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=bundle.name)
    shutil.rmtree(tree)
    shutil.rmtree(staging)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slim", action="store_true", help="build the PyPI artifacts")
    parser.add_argument(
        "--offline", action="store_true", help="build the offline bundle"
    )
    parser.add_argument("--all", action="store_true", help="build both (the default)")
    parser.add_argument(
        "--dist",
        default=str(REPO_ROOT / "dist"),
        help=(
            "output directory for the PyPI artifacts (default: ./dist). "
            "Kept free of anything twine must not see, so "
            "'twine check dist/*' is always safe."
        ),
    )
    parser.add_argument(
        "--offline-dist",
        default=str(REPO_ROOT / "dist-offline"),
        help=(
            "output directory for the offline bundle archive "
            "(default: ./dist-offline) — deliberately *not* ./dist"
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "scratch directory for the throwaway build copies (default: "
            "<offline-dist>/.build). Needs ~2.5 GB for the offline bundle."
        ),
    )
    parser.add_argument(
        "--model-source",
        default=None,
        help=(
            "Hugging Face cache root (or single model directory) to "
            "provision the embedded models from instead of downloading, "
            "e.g. ~/.local/share/corvidex/embed-cache"
        ),
    )
    parser.add_argument(
        "--models-offline",
        action="store_true",
        help="never download a model: one missing from --model-source is an error",
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="interpreter the wheelhouse targets (default: 3.12)",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=None,
        help=(
            "pip platform tag for the wheelhouse, repeatable. pip matches "
            "these literally against wheel tags (it does not expand a "
            "manylinux version into the lower ones), so the default asks "
            f"for the union {DEFAULT_PLATFORMS}; the resulting wheelhouse's "
            "real floor is the strictest tag pip ends up picking."
        ),
    )
    parser.add_argument(
        "--platform-label",
        default=None,
        help="platform name in the bundle filename (default: from --platform)",
    )
    args = parser.parse_args()

    slim = args.slim or args.all or not (args.slim or args.offline)
    offline = args.offline or args.all or not (args.slim or args.offline)

    dist = Path(args.dist).expanduser().resolve()
    offline_dist = Path(args.offline_dist).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else offline_dist / ".build"
    )
    model_source = (
        Path(args.model_source).expanduser().resolve() if args.model_source else None
    )
    platforms = args.platform or DEFAULT_PLATFORMS
    platform_label = args.platform_label or platforms[0].replace("_", "-")
    abi_label = f"cp{args.python_version.replace('.', '')}"

    dist.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    if slim:
        produced += build_slim(work_dir, dist)
    if offline:
        produced.append(
            build_offline(
                work_dir,
                offline_dist,
                model_source,
                args.models_offline,
                args.python_version,
                platforms,
                platform_label,
                abi_label,
            )
        )

    print("\n== artifacts ==")
    for path in produced:
        print(f"  {human(path.stat().st_size):>10}  {path}")
    if slim:
        print("\nPyPI-ready (slim) artifacts are in dist/; check them with:")
        print("  uv run --no-sync --with twine twine check dist/*.whl dist/*.tar.gz")
        print("Nothing is uploaded by this script.")


if __name__ == "__main__":
    main()
