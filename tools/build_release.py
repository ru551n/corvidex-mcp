"""Build the distribution artifacts: the PyPI wheel and the offline set.

They cannot be one artifact. PyPI enforces a 100 MB per-file limit (and
a per-project quota), and limit increases are not granted for bundled
model weights — the three models this server loads are ~0.86 GB of ONNX
weights. So:

* **slim** — ``dist/corvidex_mcp-<version>-py3-none-any.whl`` plus the
  sdist: no ``assets/*/onnx/`` weights, models fetched on first run
  (the current default behaviour). This is what ``twine upload`` gets.

* **offline** — ``dist-offline/``: an all-models wheel plus one
  dependency wheelhouse per supported platform/interpreter. Distributed
  outside PyPI (GitHub Release assets), for air-gapped hosts. Nothing
  from here ever lands in ``dist/``, so ``twine check dist/*`` and an
  eventual ``twine upload dist/*`` can never see it.

Splitting the weights across several sub-100 MB wheels to squeeze them
onto PyPI was considered and rejected: it abuses a shared, donated
index to host ~0.8 GB of model binaries that are not this project's
code, and it is exactly what the limit exists to prevent.

Why the offline side is *two* kinds of asset
--------------------------------------------

The package is pure Python: the all-models wheel is ``py3-none-any``
and byte-identical for every platform and interpreter. What is *not*
portable is the dependency wheelhouse — onnxruntime, tokenizers,
sqlite-vec, pydantic-core and friends ship compiled wheels per
(platform, CPython version). Shipping a self-contained bundle per
combination would republish the same 537 MB of weights a dozen times
(~7 GB per release) to vary 71 MB of dependency wheels, so the two are
split:

* ``corvidex-mcp-<version>-offline-wheel.tar`` — the shared all-models
  wheel, once per release. (It is wrapped in a tar because GitHub
  rewrites non-alphanumeric characters in release *asset* names, and
  the ``+offline`` local version segment must survive verbatim in the
  wheel filename or pip will not install it.)
* ``corvidex-mcp-<version>-wheelhouse-<target>.tar.gz`` — one per
  target: the dependency wheels, an installer, a README, and a
  machine-readable ``target.json``.

The installer takes the shared wheel plus the matching wheelhouse and
refuses, with an explicit message, to install a mismatched pair.

The two wheels must never be confusable. The embedded one carries a
PEP 440 **local version segment**, ``<version>+offline``:

* PyPI rejects local versions outright (``twine upload`` fails before
  any byte of a 0.5 GB wheel is sent), so it cannot be published by
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
        --target linux-x86_64-cp312 --target windows-x86_64-cp313
    uv run --no-sync python tools/build_release.py --list-targets
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from packaging.markers import Marker
from packaging.tags import mac_platforms

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Local version segment marking the all-models wheel (see module docstring).
OFFLINE_LOCAL_VERSION = "offline"


@dataclass(frozen=True)
class Target:
    """One (platform, CPython version) the wheelhouse is built for.

    ``platforms`` are pip ``--platform`` tags. pip matches them
    *literally* against wheel tags — it does not treat
    ``manylinux_2_28`` as satisfying a request for ``manylinux2014`` —
    so every variant a dependency may publish has to be listed, or that
    dependency silently has no candidate at all. The *effective* floor
    of the resulting wheelhouse is the strictest tag actually chosen,
    not the loosest tag requested.

    ``sys_platform``/``machines``/``os_name`` are also what the exported
    requirements' environment markers are evaluated against: pip
    evaluates markers in the *running* interpreter's environment, not
    the ``--platform`` one, so a Windows wheelhouse built on Linux
    would silently omit ``pywin32`` if we handed pip the markers
    unevaluated. :func:`target_requirements` resolves them instead.
    """

    id: str
    label: str
    python: str
    platforms: list[str]
    sys_platform: str
    os_name: str
    platform_system: str
    #: ``platform.machine()`` values accepted by the installer. The
    #: first is the canonical one for marker evaluation.
    machines: list[str]
    #: Installer flavour: ``sh`` for POSIX hosts, ``ps1`` for Windows.
    installer: str
    #: Major macOS version the wheelhouse requires, if any.
    min_macos: int | None = None
    #: Extra notes rendered into the bundle README.
    notes: list[str] = field(default_factory=list)

    @property
    def abi(self) -> str:
        return f"cp{self.python.replace('.', '')}"

    @property
    def family(self) -> str:
        """Target id without the interpreter suffix (``linux-x86_64``)."""
        return self.id.rsplit("-", 1)[0]

    def marker_environment(self) -> dict[str, str]:
        return {
            "sys_platform": self.sys_platform,
            "platform_system": self.platform_system,
            "os_name": self.os_name,
            "platform_machine": self.machines[0],
            "platform_release": "",
            "platform_version": "",
            "python_version": self.python,
            "python_full_version": f"{self.python}.0",
            "implementation_name": "cpython",
            "implementation_version": f"{self.python}.0",
            "platform_python_implementation": "CPython",
            "extra": "",
        }


#: glibc floors a Linux wheelhouse may draw from. manylinux2014 is the
#: alias of manylinux_2_17; both spellings occur in the wild.
_LINUX_TAGS = ["manylinux2014", "manylinux_2_17", "manylinux_2_28", "manylinux_2_34"]

#: Apple Silicon deployment target. One tag is enough: unlike
#: manylinux, pip *does* expand a macOS ``--platform`` request downwards
#: (and across binary formats), so ``macosx_14_0_arm64`` also accepts
#: the ``macosx_11_0_arm64`` and ``macosx_10_13_universal2`` wheels most
#: dependencies publish. 14 is where it has to sit: onnxruntime ships
#: its arm64 macOS wheels only as ``macosx_14_0_arm64``, for every
#: CPython version, so no lower floor resolves at all.
_MACOS_TAG = "macosx_14_0_arm64"
_MACOS_FLOOR = 14


def _linux(arch: str, python: str) -> Target:
    return Target(
        id=f"linux-{arch}-cp{python.replace('.', '')}",
        label=f"Linux {arch} (glibc 2.34+)",
        python=python,
        platforms=[f"{tag}_{arch}" for tag in _LINUX_TAGS],
        sys_platform="linux",
        os_name="posix",
        platform_system="Linux",
        machines=[arch],
        installer="sh",
        notes=[
            "The strictest manylinux tag pip selects here is glibc 2.34, "
            "i.e. RHEL 9 / Ubuntu 22.04 and newer.",
        ],
    )


def _macos(python: str) -> Target:
    return Target(
        id=f"macos-arm64-cp{python.replace('.', '')}",
        label=f"macOS {_MACOS_FLOOR}+ on Apple Silicon (arm64)",
        python=python,
        platforms=[_MACOS_TAG],
        sys_platform="darwin",
        os_name="posix",
        platform_system="Darwin",
        machines=["arm64"],
        installer="sh",
        min_macos=_MACOS_FLOOR,
        notes=[
            "macOS 14 (Sonoma) or newer is required: the pinned "
            "onnxruntime publishes its arm64 wheels only as "
            "`macosx_14_0_arm64`, for every CPython version. Intel Macs "
            "are not covered - onnxruntime has dropped x86_64 macOS "
            "wheels entirely.",
        ],
    )


def _windows(python: str) -> Target:
    return Target(
        id=f"windows-x86_64-cp{python.replace('.', '')}",
        label="Windows x86-64",
        python=python,
        platforms=["win_amd64"],
        sys_platform="win32",
        os_name="nt",
        platform_system="Windows",
        machines=["AMD64", "x86_64"],
        installer="ps1",
        notes=[
            "Install with `install.ps1` from PowerShell; `install.sh` is "
            "not shipped for this target.",
        ],
    )


#: Every target a release ships a wheelhouse for. Each one is *verified*
#: at build time: the wheelhouse download must resolve completely and
#: every wheel in it must carry a tag this target asked for
#: (:func:`verify_wheelhouse`), so a target that stops resolving fails
#: the build instead of producing an unusable archive.
TARGETS: list[Target] = [
    *(_linux("x86_64", py) for py in ("3.12", "3.13", "3.14")),
    *(_linux("aarch64", py) for py in ("3.12", "3.13", "3.14")),
    *(_macos(py) for py in ("3.12", "3.13", "3.14")),
    *(_windows(py) for py in ("3.12", "3.13", "3.14")),
]

TARGETS_BY_ID = {target.id: target for target in TARGETS}

#: Combinations that are deliberately **not** built, with the reason.
#: Listed rather than silently absent so ``--list-targets`` and the docs
#: answer "why is there no asset for my machine?" without a bisect
#: through PyPI.
UNSUPPORTED: dict[str, str] = {
    "macos-x86_64-*": (
        "Intel macOS: onnxruntime publishes no x86_64 macOS wheels for "
        "the pinned version, at any CPython version."
    ),
    "windows-x86_64-cp313t": (
        "Free-threaded Windows: pywin32 (a dependency of mcp) publishes "
        "no cp313t/cp314t wheels, so no free-threaded Windows "
        "wheelhouse can ever resolve."
    ),
    "windows-x86_64-cp314t": (
        "Free-threaded Windows: pywin32 (a dependency of mcp) publishes "
        "no cp313t/cp314t wheels, so no free-threaded Windows "
        "wheelhouse can ever resolve."
    ),
    "*-cp3XXt (free-threaded)": (
        "Free-threaded CPython in general: the dependency set is not "
        "fully published for it, and the installers reject a "
        "free-threaded interpreter rather than install wheels built for "
        "the GIL build."
    ),
}

#: Paths never copied into a build tree (build output, caches, VCS).
COPY_EXCLUDES = shutil.ignore_patterns(
    ".git",
    ".venv",
    "dist",
    "dist-offline",
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


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def canonical(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


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


# --------------------------------------------------------------------------
# wheelhouse
# --------------------------------------------------------------------------


def export_requirements(tree: Path, dest: Path) -> str:
    """The locked runtime dependency set, markers *not* yet evaluated."""
    with dest.open("w", encoding="utf-8") as fh:
        subprocess.run(
            ["uv", "export", "--no-hashes", "--no-emit-project", "--no-dev"],
            cwd=tree,
            check=True,
            stdout=fh,
        )
    return dest.read_text(encoding="utf-8")


def target_requirements(exported: str, target: Target) -> list[str]:
    """``exported`` with every environment marker resolved for ``target``.

    pip evaluates markers against the interpreter it is *running on*,
    never against ``--platform``/``--python-version``. Handing it the
    markers verbatim from a Linux runner would drop ``pywin32`` from
    every Windows wheelhouse (and keep it out of none), which only
    surfaces as a failed install on the air-gapped host. So they are
    resolved here and stripped.
    """
    environment = target.marker_environment()
    kept: list[str] = []
    for raw in exported.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):  # --index-url and friends: pass through
            kept.append(line)
            continue
        spec, separator, marker = line.partition(";")
        if separator and not Marker(marker.strip()).evaluate(environment):
            continue
        kept.append(spec.strip())
    return kept


def build_wheelhouse(
    tree: Path, dest: Path, target: Target, requirements: Path
) -> None:
    """Download every runtime dependency for ``target`` as a wheel.

    Binary-only and pinned to the target interpreter/platform, so the
    air-gapped host never has to build anything (it has no index and
    quite possibly no compiler). ``--abi``/``--implementation`` are
    explicit because pip otherwise derives the ABI from the interpreter
    it happens to be running on — under a free-threaded CPython that
    means a ``cp314t`` ABI and *nothing* resolves.
    """
    dest.mkdir(parents=True, exist_ok=True)
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
        "--implementation",
        "cp",
        "--python-version",
        target.python,
        "--abi",
        target.abi,
    ]
    for platform in target.platforms:
        cmd += ["--platform", platform]
    run(cmd, cwd=tree)


def expand_platform_tag(requested: str) -> set[str]:
    """Every wheel platform tag pip accepts for ``--platform requested``.

    pip does not treat ``--platform`` uniformly. A macOS request is
    expanded downwards and across binary formats (so
    ``macosx_14_0_arm64`` also accepts ``macosx_11_0_arm64`` and
    ``macosx_10_13_universal2``), and the legacy ``manylinux2014`` /
    ``manylinux2010`` spellings pull in their older aliases — but the
    PEP 600 ``manylinux_x_y`` names are matched literally, which is why
    :data:`_LINUX_TAGS` still has to list each of them. Mirrored here so
    :func:`verify_wheelhouse` can tell "pip picked a tag we asked for"
    from "pip picked a tag that only works on the build host".
    """
    if requested.startswith("macosx_"):
        _, major, minor, arch = requested.split("_", 3)
        return set(mac_platforms((int(major), int(minor)), arch))
    prefix, separator, suffix = requested.partition("_")
    if prefix == "manylinux2014" and suffix in {"i686", "x86_64"}:
        return {
            requested,
            f"manylinux2010{separator}{suffix}",
            f"manylinux1{separator}{suffix}",
        }
    if prefix == "manylinux2010":
        return {requested, f"manylinux1{separator}{suffix}"}
    return {requested}


def verify_wheelhouse(
    dest: Path, requirements: list[str], target: Target
) -> list[Path]:
    """Fail loudly if the wheelhouse is not exactly what ``target`` needs.

    pip already errors out when a requirement has no candidate, but that
    only covers the resolution; this also catches a wheel that resolved
    to a tag the target never asked for (which would install on the
    build host and nowhere else).
    """
    wheels = sorted(dest.glob("*.whl"))
    strays = [p.name for p in dest.iterdir() if p.suffix != ".whl"]
    if strays:
        raise SystemExit(f"{target.id}: non-wheel files in the wheelhouse: {strays}")

    expected = {
        canonical(re.split(r"[=<>!~\[ ;]", spec, maxsplit=1)[0])
        for spec in requirements
        if not spec.startswith("-")
    }
    found = {canonical(path.name.split("-")[0]) for path in wheels}
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(
            f"{target.id}: wheelhouse is missing {len(missing)} requirement(s): "
            f"{missing}"
        )

    allowed_platforms = {"any"}
    for requested in target.platforms:
        allowed_platforms |= expand_platform_tag(requested)
    allowed_abis = {"none", "abi3", target.abi}
    for wheel in wheels:
        parts = wheel.stem.split("-")
        platform_tag, abi_tag = parts[-1], parts[-2]
        if not set(platform_tag.split(".")) & allowed_platforms:
            raise SystemExit(
                f"{target.id}: {wheel.name} carries platform tag "
                f"{platform_tag!r}, which this target never requested"
            )
        if not set(abi_tag.split(".")) & allowed_abis:
            raise SystemExit(
                f"{target.id}: {wheel.name} carries ABI tag {abi_tag!r}, "
                f"expected one of {sorted(allowed_abis)}"
            )
    print(f"  verified: {len(wheels)} wheels, all tagged for {target.id}")
    return wheels


# --------------------------------------------------------------------------
# installers
# --------------------------------------------------------------------------

INSTALL_SH = """\
#!/bin/sh
# Offline installer for corvidex-mcp @@VERSION@@ --- @@TARGET_ID@@.
# No network access is used or needed.
#
# It needs two release assets:
#   corvidex-mcp-@@VERSION@@-offline-wheel.tar          (shared, every target)
#   corvidex-mcp-@@VERSION@@-wheelhouse-@@TARGET_ID@@.tar.gz  (this one)
#
# Usage: ./install.sh [VENV_DIR] [--wheel PATH]
set -eu

TARGET_ID="@@TARGET_ID@@"
TARGET_FAMILY="@@TARGET_FAMILY@@"
TARGET_LABEL="@@TARGET_LABEL@@"
TARGET_PY="@@TARGET_PY@@"
TARGET_SYS_PLATFORM="@@SYS_PLATFORM@@"
TARGET_MACHINES="@@MACHINES@@"
TARGET_MIN_MACOS="@@MIN_MACOS@@"
WHEEL_NAME="corvidex_mcp-@@VERSION@@+offline-py3-none-any.whl"
VERSION="@@VERSION@@"

here=$(cd "$(dirname "$0")" && pwd)
venv=""
wheel="${CORVIDEX_WHEEL:-}"

usage() {
    cat <<USAGE
usage: install.sh [VENV_DIR] [--wheel PATH]

  VENV_DIR       where to create the virtualenv (default: ./corvidex-venv)
  --wheel PATH   the shared all-models wheel. Default: $WHEEL_NAME
                 looked up next to this directory, then inside it, then
                 in the current directory. CORVIDEX_WHEEL also works.
  PYTHON=...     interpreter to use (default: python$TARGET_PY)

This wheelhouse is built for CPython $TARGET_PY on $TARGET_LABEL and
installs on nothing else; see README.md for the other targets.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --wheel) [ $# -ge 2 ] || { echo "error: --wheel needs a path" >&2; exit 2; }
                 wheel="$2"; shift 2 ;;
        --wheel=*) wheel="${1#--wheel=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) if [ -n "$venv" ]; then
               echo "error: unexpected argument: $1" >&2; usage >&2; exit 2
           fi
           venv="$1"; shift ;;
    esac
done
[ -n "$venv" ] || venv="$(pwd)/corvidex-venv"

# --- the shared all-models wheel ---------------------------------------
if [ -z "$wheel" ]; then
    for candidate in "$here/../$WHEEL_NAME" "$here/$WHEEL_NAME" \\
                     "$(pwd)/$WHEEL_NAME"; do
        if [ -f "$candidate" ]; then wheel="$candidate"; break; fi
    done
fi
if [ -z "$wheel" ] || [ ! -f "$wheel" ]; then
    cat >&2 <<MISSING
error: the all-models wheel was not found.

  looked for: $WHEEL_NAME
      beside: $here/..
      inside: $here
          in: $(pwd)

It is a separate release asset, shared by every target. Fetch and
unpack it next to this directory:

  tar xf corvidex-mcp-$VERSION-offline-wheel.tar

or point at it directly:  ./install.sh "$venv" --wheel /path/to/$WHEEL_NAME
MISSING
    exit 1
fi

# --- the interpreter ----------------------------------------------------
python="${PYTHON:-python$TARGET_PY}"
if ! command -v "$python" >/dev/null 2>&1; then
    echo "error: interpreter '$python' not found on PATH." >&2
    echo "       This wheelhouse needs CPython $TARGET_PY; set" >&2
    echo "       PYTHON=/path/to/python$TARGET_PY and retry." >&2
    exit 1
fi

probe=$("$python" - <<'PY'
import platform, sys, sysconfig
print("%d.%d" % sys.version_info[:2])
print(sys.platform)
print(platform.machine() or "unknown")
print("free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "gil")
print(platform.python_implementation())
print((platform.mac_ver()[0] or "0").split(".")[0])
PY
) || { echo "error: could not run '$python'." >&2; exit 1; }
# shellcheck disable=SC2086
set -- $probe
have_py="$1"; have_sys="$2"; have_machine="$3"
have_threading="$4"; have_impl="$5"; have_macos="$6"

mismatch() {
    echo "error: this wheelhouse does not match '$python'." >&2
    echo >&2
    echo "  wheelhouse:  $TARGET_ID (CPython $TARGET_PY)" >&2
    echo "               $TARGET_LABEL" >&2
    echo "  interpreter: $have_impl $have_py, $have_threading build" >&2
    echo "               on $have_sys/$have_machine" >&2
    echo >&2
    echo "  $1" >&2
    echo >&2
    echo "Installing anyway would put wheels built for another" >&2
    echo "platform or ABI into the venv, so this stops here." >&2
    exit 1
}

want_cp="cp$(echo "$have_py" | tr -d .)"
other="corvidex-mcp-$VERSION-wheelhouse-$TARGET_FAMILY-$want_cp.tar.gz"

if [ "$have_impl" != "CPython" ]; then
    mismatch "Only CPython is supported; found $have_impl."
fi
if [ "$have_threading" != "gil" ]; then
    mismatch "These wheels are for the GIL build, not free-threaded CPython."
fi
if [ "$have_py" != "$TARGET_PY" ]; then
    mismatch "Download $other instead, or retry with PYTHON=python$TARGET_PY."
fi
if [ "$have_sys" != "$TARGET_SYS_PLATFORM" ]; then
    mismatch "Wrong OS: these wheels are for '$TARGET_SYS_PLATFORM'."
fi
case " $TARGET_MACHINES " in
    *" $have_machine "*) ;;
    *) mismatch "Wrong CPU: these wheels are for [$TARGET_MACHINES]." ;;
esac
if [ -n "$TARGET_MIN_MACOS" ] && [ "$have_macos" -lt "$TARGET_MIN_MACOS" ]; then
    mismatch "macOS $TARGET_MIN_MACOS+ is required (found macOS $have_macos)."
fi

# --- install ------------------------------------------------------------
echo "creating venv: $venv"
"$python" -m venv "$venv"

echo "installing from the bundled wheels (no index)"
"$venv/bin/pip" install --no-index --find-links "$here/wheelhouse" "$wheel"

echo
version=$("$venv/bin/pip" show corvidex-mcp | sed -n 's/^Version: //p')
echo "installed:  corvidex-mcp $version  ($TARGET_ID)"
echo "server:     $venv/bin/corvidex-mcp"
echo
echo "verify (still offline):"
echo "  $venv/bin/python -m corvidex_mcp.verify_offline"
echo
echo "register with an MCP client, e.g.:"
echo "  claude mcp add corvidex-mcp -- $venv/bin/corvidex-mcp"
"""


INSTALL_PS1 = """\
# Offline installer for corvidex-mcp @@VERSION@@ --- @@TARGET_ID@@.
# No network access is used or needed.
#
# It needs two release assets:
#   corvidex-mcp-@@VERSION@@-offline-wheel.tar          (shared, every target)
#   corvidex-mcp-@@VERSION@@-wheelhouse-@@TARGET_ID@@.tar.gz  (this one)
#
# Usage: .\\install.ps1 [-VenvDir <path>] [-Wheel <path>] [-Python <exe>]
[CmdletBinding()]
param(
    [string] $VenvDir = (Join-Path (Get-Location) 'corvidex-venv'),
    [string] $Wheel = $env:CORVIDEX_WHEEL,
    [string] $Python = ''
)

$ErrorActionPreference = 'Stop'

$TargetId       = '@@TARGET_ID@@'
$TargetFamily   = '@@TARGET_FAMILY@@'
$TargetLabel    = '@@TARGET_LABEL@@'
$TargetPy       = '@@TARGET_PY@@'
$TargetSys      = '@@SYS_PLATFORM@@'
$TargetMachines = '@@MACHINES@@'.Split(' ')
$Version        = '@@VERSION@@'
$WheelName      = "corvidex_mcp-$Version+offline-py3-none-any.whl"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- the shared all-models wheel ---------------------------------------
if (-not $Wheel) {
    foreach ($candidate in @(
        (Join-Path (Split-Path -Parent $here) $WheelName),
        (Join-Path $here $WheelName),
        (Join-Path (Get-Location) $WheelName)
    )) {
        if (Test-Path -LiteralPath $candidate) { $Wheel = $candidate; break }
    }
}
if (-not $Wheel -or -not (Test-Path -LiteralPath $Wheel)) {
    Write-Error @"
the all-models wheel was not found.

  looked for: $WheelName
      beside: $(Split-Path -Parent $here)
      inside: $here
          in: $(Get-Location)

It is a separate release asset, shared by every target. Fetch and
unpack it next to this directory:

  tar xf corvidex-mcp-$Version-offline-wheel.tar

or point at it directly:  .\\install.ps1 -Wheel C:\\path\\to\\$WheelName
"@
}

# --- the interpreter ----------------------------------------------------
if (-not $Python) {
    $launcher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($launcher) {
        $Python = "py -$TargetPy"
    } else {
        $Python = 'python'
    }
}
$pythonParts = $Python.Split(' ')
$pythonExe = $pythonParts[0]
$pythonArgs = @()
if ($pythonParts.Length -gt 1) {
    $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)]
}

if (-not (Get-Command $pythonExe -ErrorAction SilentlyContinue)) {
    Write-Error @"
interpreter '$pythonExe' not found on PATH.
This wheelhouse needs CPython $TargetPy; pass
-Python C:\\path\\to\\python.exe
"@
}

$probeScript = @'
import platform, sys, sysconfig
print("%d.%d" % sys.version_info[:2])
print(sys.platform)
print(platform.machine() or "unknown")
print("free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "gil")
print(platform.python_implementation())
'@
$probeFile = Join-Path ([System.IO.Path]::GetTempPath()) "corvidex-probe-$PID.py"
Set-Content -LiteralPath $probeFile -Value $probeScript -Encoding ascii
try {
    $probe = & $pythonExe @pythonArgs $probeFile
} finally {
    Remove-Item -LiteralPath $probeFile -ErrorAction SilentlyContinue
}
if ($LASTEXITCODE -ne 0) { Write-Error "could not run '$Python'." }

$havePy, $haveSys, $haveMachine, $haveThreading, $haveImpl = $probe

function Stop-Mismatch([string] $why) {
    Write-Error @"
this wheelhouse does not match '$Python'.

  wheelhouse:  $TargetId (CPython $TargetPy)
               $TargetLabel
  interpreter: $haveImpl $havePy, $haveThreading build
               on $haveSys/$haveMachine

  $why

Installing anyway would put wheels built for another platform or ABI
into the venv, so this stops here.
"@
}

$want = 'cp' + $havePy.Replace('.', '')
$other = "corvidex-mcp-$Version-wheelhouse-$TargetFamily-$want.tar.gz"
$machines = $TargetMachines -join ', '

if ($haveImpl -ne 'CPython') {
    Stop-Mismatch "Only CPython is supported; found $haveImpl."
}
if ($haveThreading -ne 'gil') {
    Stop-Mismatch @"
These wheels are for the GIL build, not free-threaded CPython.
  pywin32 (a dependency of mcp) publishes no free-threaded wheels,
  so no free-threaded Windows wheelhouse can exist at all.
"@
}
if ($havePy -ne $TargetPy) {
    Stop-Mismatch "Download $other instead, or pass -Python 'py -$TargetPy'."
}
if ($haveSys -ne $TargetSys) {
    Stop-Mismatch "Wrong OS: these wheels are for '$TargetSys'."
}
if ($TargetMachines -notcontains $haveMachine) {
    Stop-Mismatch "Wrong CPU: these wheels are for [$machines]."
}

# --- install ------------------------------------------------------------
Write-Host "creating venv: $VenvDir"
& $pythonExe @pythonArgs -m venv $VenvDir
if ($LASTEXITCODE -ne 0) { Write-Error 'venv creation failed.' }

$pip = Join-Path $VenvDir 'Scripts\\pip.exe'
Write-Host 'installing from the bundled wheels (no index)'
& $pip install --no-index --find-links (Join-Path $here 'wheelhouse') $Wheel
if ($LASTEXITCODE -ne 0) { Write-Error 'offline install failed.' }

$line = (& $pip show corvidex-mcp | Select-String '^Version: ').ToString()
$installed = $line.Substring(9)
$venvPython = Join-Path $VenvDir 'Scripts\\python.exe'
$server = Join-Path $VenvDir 'Scripts\\corvidex-mcp.exe'
Write-Host ''
Write-Host "installed:  corvidex-mcp $installed  ($TargetId)"
Write-Host "server:     $server"
Write-Host ''
Write-Host 'verify (still offline):'
Write-Host "  $venvPython -m corvidex_mcp.verify_offline"
Write-Host ''
Write-Host 'register with an MCP client, e.g.:'
Write-Host "  claude mcp add corvidex-mcp -- $server"
"""


def render_installer(template: str, *, target: Target, version: str) -> str:
    replacements = {
        "@@VERSION@@": version,
        "@@TARGET_ID@@": target.id,
        "@@TARGET_FAMILY@@": target.family,
        "@@TARGET_LABEL@@": target.label,
        "@@TARGET_PY@@": target.python,
        "@@SYS_PLATFORM@@": target.sys_platform,
        "@@MACHINES@@": " ".join(target.machines),
        "@@MIN_MACOS@@": "" if target.min_macos is None else str(target.min_macos),
    }
    for needle, value in replacements.items():
        template = template.replace(needle, value)
    if "@@" in template:
        raise SystemExit(f"unreplaced placeholder in the {target.id} installer")
    return template


# --------------------------------------------------------------------------
# bundle docs
# --------------------------------------------------------------------------

WHEELHOUSE_README = """\
# corvidex-mcp {version} — offline wheelhouse for {target_label}

Half of an air-gapped install. The other half is the **shared
all-models wheel**, which is platform-independent and published once
per release:

| Asset | Size | Needed by |
| --- | --- | --- |
| `corvidex-mcp-{version}-offline-wheel.tar` | {wheel_size} | every target |
| `...-{version}-wheelhouse-{target_id}.tar.gz` | {archive_size} | this target |

The package is pure Python, so its wheel (`py3-none-any`, every model
embedded) is identical everywhere; only the compiled dependency wheels
in `wheelhouse/` are specific to a platform and a CPython version.
Shipping them separately keeps a release at roughly one copy of the
weights instead of one per target.

## This wheelhouse targets

* **{target_label}**
* **CPython {python_version}**, the standard GIL build
* wheel tags requested: {platform_tags}

`install.sh` checks the interpreter against all of that before it
installs anything, so pairing this archive with the wrong Python or
the wrong machine fails with a message instead of a half-broken venv.

{notes}

## Contents

* `wheelhouse/` — every runtime dependency as a wheel
  ({wheelhouse_count} files, {wheelhouse_size})
* `{installer}` — creates a venv and installs, with `--no-index`
* `target.json` — the same target metadata, machine-readable
* `README.md` — this file

## Install

```console
$ tar xf corvidex-mcp-{version}-offline-wheel.tar
$ tar xzf corvidex-mcp-{version}-wheelhouse-{target_id}.tar.gz
$ cd corvidex-mcp-{version}-wheelhouse-{target_id}
$ {install_command}
```

The installer looks for `{wheel_name}` next to this directory first
(which is where the command above unpacks it), then inside it, then in
the current directory; `{wheel_flag}` points at it anywhere else.

## Verify (offline)

```console
$ {verify_command}
```

It reports which models are loaded from the bundled assets, indexes a
throwaway repository through the real pipeline, and runs a search over
all three collections with reranking on — with every outbound socket
refused, so a silent download would fail rather than hide. Expect a
final `offline verification: OK` line.

## Register with an MCP client

```console
$ claude mcp add corvidex-mcp -- {server_path}
```

The server indexes the directory it is started in; see
`docs/configuration.md` in the repository for `.corvidex` options.

## Requirements on the target host

* CPython {python_version} with `venv` and `pip`
* `git` (repositories are indexed from Git working trees)
* `vhdl_ls` / Veridian are **optional** — without them HDL files fall
  back to structural parsing.

## About the version number

The wheel is `{marked_version}`, not `{version}`: the `+offline` suffix
is a PEP 440 local version segment. It marks this build as the
all-models variant (`pip show corvidex-mcp` reports it), it compares
higher than the plain `{version}` on PyPI so an offline host is never
silently downgraded, and PyPI refuses local versions outright, so this
wheel cannot be uploaded there by mistake.

This is **not on PyPI** and never will be: PyPI enforces a 100 MB
per-file limit and does not grant increases for bundled model weights,
and the embedded models are {weights_size} of ONNX weights. PyPI
carries the slim wheel (`pip install corvidex-mcp`, models downloaded
on first run); this pair is distributed as GitHub Release assets
instead.

## What is in the shared wheel

| Model | Role | Weights |
| --- | --- | --- |
{model_rows}

They live in `corvidex_mcp/assets/<org>-<model>/` inside the wheel and
are loaded from there directly (fastembed `specific_model_path`), so no
Hugging Face cache lookup and no download ever happens.
"""


def write_wheelhouse_docs(
    bundle: Path,
    *,
    target: Target,
    version: str,
    marked_version: str,
    wheel_name: str,
    wheel_size: int,
    archive_size_hint: str,
    models: list[tuple[str, str, int]],
) -> None:
    wheelhouse = bundle / "wheelhouse"
    wheelhouse_files = sorted(wheelhouse.glob("*"))
    posix = target.installer == "sh"
    installer = "install.sh" if posix else "install.ps1"
    install_command = (
        "./install.sh /opt/corvidex-venv"
        if posix
        else ".\\install.ps1 -VenvDir C:\\corvidex-venv"
    )
    verify_command = (
        "/opt/corvidex-venv/bin/python -m corvidex_mcp.verify_offline"
        if posix
        else "C:\\corvidex-venv\\Scripts\\python.exe -m corvidex_mcp.verify_offline"
    )
    server_path = (
        "/opt/corvidex-venv/bin/corvidex-mcp"
        if posix
        else "C:\\corvidex-venv\\Scripts\\corvidex-mcp.exe"
    )
    notes = "\n\n".join(f"> [!NOTE]\n> {note}" for note in target.notes)

    (bundle / "README.md").write_text(
        WHEELHOUSE_README.format(
            version=version,
            marked_version=marked_version,
            target_id=target.id,
            target_label=target.label,
            python_version=target.python,
            platform_tags=", ".join(f"`{tag}`" for tag in target.platforms),
            notes=notes,
            installer=installer,
            install_command=install_command,
            verify_command=verify_command,
            server_path=server_path,
            wheel_name=wheel_name,
            wheel_flag="--wheel" if posix else "-Wheel",
            wheel_size=human(wheel_size),
            archive_size=archive_size_hint,
            wheelhouse_count=len(wheelhouse_files),
            wheelhouse_size=human(sum(p.stat().st_size for p in wheelhouse_files)),
            weights_size=human(sum(size for _, _, size in models)),
            model_rows="\n".join(
                f"| `{name}` | {role} | {human(size)} |" for name, role, size in models
            ),
        ),
        encoding="utf-8",
    )

    (bundle / "target.json").write_text(
        json.dumps(
            {
                "target": target.id,
                "label": target.label,
                "python_version": target.python,
                "abi": target.abi,
                "free_threaded": False,
                "sys_platform": target.sys_platform,
                "machines": target.machines,
                "platform_tags": target.platforms,
                "min_macos": target.min_macos,
                "corvidex_mcp_version": marked_version,
                "wheel": wheel_name,
                "wheelhouse_wheels": len(wheelhouse_files),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    script = bundle / installer
    script.write_text(
        render_installer(
            INSTALL_SH if posix else INSTALL_PS1, target=target, version=version
        ),
        encoding="utf-8",
    )
    if posix:
        script.chmod(0o755)


# --------------------------------------------------------------------------
# the offline build
# --------------------------------------------------------------------------


def collect_model_metadata(tree: Path) -> list[tuple[str, str, int]]:
    assets = tree / "src" / "corvidex_mcp" / "assets"
    metadata: list[tuple[str, str, int]] = []
    sys.path.insert(0, str(tree / "src"))
    try:
        from corvidex_mcp.embeddings.assets import MODEL_WEIGHTS, required_models

        for model in required_models():
            weights = assets / model.subdir / MODEL_WEIGHTS
            metadata.append(
                (model.name, ", ".join(model.used_by), weights.stat().st_size)
            )
    finally:
        sys.path.pop(0)
    return metadata


def build_offline(
    work_dir: Path,
    offline_dist: Path,
    model_source: Path | None,
    models_offline: bool,
    targets: list[Target],
) -> list[Path]:
    print("== offline: all-models wheel (shared by every target) ==")
    tree = fresh_copy(work_dir, "offline")
    version = project_version(tree)
    marked = set_local_version(tree, OFFLINE_LOCAL_VERSION)
    provision_models(tree, model_source, models_offline)
    models = collect_model_metadata(tree)

    offline_dist.mkdir(parents=True, exist_ok=True)
    staging = work_dir / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (wheel,) = uv_build(tree, staging / "wheel", wheel_only=True)
    (staging / "wheel" / ".gitignore").unlink(missing_ok=True)  # uv's, not ours

    # Uncompressed on purpose: the wheel is already a deflated zip, so
    # gzip would spend minutes to save nothing. The tar exists only so
    # that the '+' in the wheel filename survives GitHub's release-asset
    # name rewriting; pip refuses a wheel whose filename no longer
    # matches its metadata.
    wheel_archive = offline_dist / f"corvidex-mcp-{version}-offline-wheel.tar"
    wheel_archive.unlink(missing_ok=True)
    with tarfile.open(wheel_archive, "w") as tar:
        tar.add(wheel, arcname=wheel.name)
    produced = [wheel_archive]

    exported = export_requirements(tree, staging / "requirements-bundle.txt")

    for index, target in enumerate(targets, start=1):
        print(f"\n== offline: wheelhouse {index}/{len(targets)} — {target.id} ==")
        requirements = target_requirements(exported, target)
        req_file = staging / f"requirements-{target.id}.txt"
        req_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        bundle = staging / f"corvidex-mcp-{version}-wheelhouse-{target.id}"
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        build_wheelhouse(tree, bundle / "wheelhouse", target, req_file)
        verify_wheelhouse(bundle / "wheelhouse", requirements, target)

        archive = offline_dist / f"{bundle.name}.tar.gz"
        write_wheelhouse_docs(
            bundle,
            target=target,
            version=version,
            marked_version=marked,
            wheel_name=wheel.name,
            wheel_size=wheel.stat().st_size,
            # The archive does not exist yet; the wheelhouse is what
            # dominates it, and gzip barely moves already-zipped wheels.
            archive_size_hint=f"~{human(dir_size(bundle))}",
            models=models,
        )
        archive.unlink(missing_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(bundle, arcname=bundle.name)
        # Reclaim before the next target: twelve wheelhouses at once is
        # ~0.9 GB of scratch nobody needs.
        shutil.rmtree(bundle)
        produced.append(archive)

    shutil.rmtree(tree)
    shutil.rmtree(staging)
    return produced


def assert_dist_is_pypi_safe(dist: Path) -> None:
    """``dist/`` must never hold an all-models wheel.

    The one mistake that would matter is an ``+offline`` wheel sitting
    where ``twine upload dist/*`` would find it.
    """
    leaked = sorted(p.name for p in dist.glob("*") if OFFLINE_LOCAL_VERSION in p.name)
    if leaked:
        raise SystemExit(
            f"refusing to finish: +offline artifacts leaked into {dist}: {leaked}"
        )


def resolve_targets(args: argparse.Namespace) -> list[Target]:
    """The targets to build, from ``--target`` or the ad-hoc flags."""
    ad_hoc = args.platform or args.platform_label or args.python_version
    if args.target and ad_hoc:
        raise SystemExit(
            "--target and --platform/--platform-label/--python-version are "
            "alternatives: --target picks from the supported matrix, the "
            "others define one ad-hoc target."
        )
    if ad_hoc:
        return [ad_hoc_target(args)]
    if not args.target:
        return list(TARGETS)

    chosen: list[Target] = []
    for name in args.target:
        if name == "all":
            chosen = list(TARGETS)
            break
        if name in TARGETS_BY_ID:
            chosen.append(TARGETS_BY_ID[name])
            continue
        family = [t for t in TARGETS if t.family == name]
        if family:
            chosen.extend(family)
            continue
        reason = UNSUPPORTED.get(name)
        if reason:
            raise SystemExit(f"target {name!r} is not built: {reason}")
        raise SystemExit(
            f"unknown target {name!r}. Known: "
            f"{', '.join(sorted(TARGETS_BY_ID))}. Use --list-targets."
        )
    chosen_ids = {target.id for target in chosen}
    return [target for target in TARGETS if target.id in chosen_ids]


def ad_hoc_target(args: argparse.Namespace) -> Target:
    """A one-off target from ``--platform``/``--python-version``.

    Keeps the pre-matrix invocation working for a host the matrix does
    not cover (musl, an older glibc, win_arm64, ...). Everything the
    marker environment needs is inferred from the first platform tag.
    """
    python = args.python_version or "3.12"
    platforms = args.platform or [f"{tag}_x86_64" for tag in _LINUX_TAGS]
    first = platforms[0]
    if first.startswith("win"):
        os_name, sys_platform, system, installer = "nt", "win32", "Windows", "ps1"
        machines = ["AMD64", "ARM64"]
    elif first.startswith("macosx"):
        os_name, sys_platform, system, installer = "posix", "darwin", "Darwin", "sh"
        machines = ["arm64", "x86_64"]
    else:
        os_name, sys_platform, system, installer = "posix", "linux", "Linux", "sh"
        machines = ["aarch64" if "aarch64" in first or "arm64" in first else "x86_64"]
    label = args.platform_label or first.replace("_", "-")
    return Target(
        id=f"{label}-cp{python.replace('.', '')}",
        label=label,
        python=python,
        platforms=platforms,
        sys_platform=sys_platform,
        os_name=os_name,
        platform_system=system,
        machines=machines,
        installer=installer,
        notes=["Built ad hoc from --platform; not part of the supported matrix."],
    )


def print_targets() -> None:
    print("Supported targets (one wheelhouse asset each):\n")
    width = max(len(t.id) for t in TARGETS)
    for target in TARGETS:
        print(f"  {target.id:<{width}}  CPython {target.python}  {target.label}")
    print("\nDeliberately not built:\n")
    for name, reason in UNSUPPORTED.items():
        print(f"  {name}\n      {reason}")
    print(
        "\nA host outside the matrix: "
        "--platform <tag> [--platform <tag> ...] --python-version X.Y"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slim", action="store_true", help="build the PyPI artifacts")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build the all-models wheel and the target wheelhouses",
    )
    parser.add_argument("--all", action="store_true", help="build both (the default)")
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="print the offline target matrix and exit",
    )
    parser.add_argument(
        "--target-ids",
        action="store_true",
        help=(
            "print one supported target id per line and exit — the "
            "machine-readable form of --list-targets, so a caller "
            "(CI) can check the artifact set without duplicating the "
            "matrix"
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="ID",
        help=(
            "offline target to build a wheelhouse for, repeatable: a full "
            "id (linux-x86_64-cp312), a family (linux-x86_64, every "
            "CPython version), or 'all'. Default: the whole matrix. "
            "See --list-targets."
        ),
    )
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
            "output directory for the offline artifacts "
            "(default: ./dist-offline) — deliberately *not* ./dist"
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "scratch directory for the throwaway build copies (default: "
            "<offline-dist>/.build). Needs ~2.5 GB for the offline build."
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
        default=None,
        metavar="X.Y",
        help=(
            "build a single ad-hoc target for this CPython version "
            "instead of the matrix (with --platform). Default when "
            "given alone: 3.12."
        ),
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=None,
        help=(
            "pip platform tag for a single ad-hoc target, repeatable. pip "
            "matches these literally against wheel tags (it does not "
            "expand a manylinux version into the lower ones), so pass "
            "every variant a dependency may publish; the resulting "
            "wheelhouse's real floor is the strictest tag pip picks."
        ),
    )
    parser.add_argument(
        "--platform-label",
        default=None,
        help="name of the ad-hoc target in its filenames (default: from --platform)",
    )
    args = parser.parse_args()

    if args.target_ids:
        for target in TARGETS:
            print(target.id)
        return
    if args.list_targets:
        print_targets()
        return

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
    targets = resolve_targets(args)

    dist.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    if slim:
        produced += build_slim(work_dir, dist)
    if offline:
        produced += build_offline(
            work_dir, offline_dist, model_source, args.models_offline, targets
        )

    assert_dist_is_pypi_safe(dist)

    print("\n== artifacts ==")
    total = 0
    for path in produced:
        size = path.stat().st_size
        total += size
        print(f"  {human(size):>10}  {path}")
    print(f"  {human(total):>10}  total")
    if slim:
        print("\nPyPI-ready (slim) artifacts are in dist/; check them with:")
        print("  uv run --no-sync --with twine twine check dist/*.whl dist/*.tar.gz")
        print("Nothing is uploaded by this script.")


if __name__ == "__main__":
    main()
