"""Simulation setup: assemble the simulation model list and the
exclude set for unisim primitives, then launch the simulator.

Unisim models are excluded from the search path for targets that ship
their own primitive library; the exclude list keeps the simulator from
resolving the wrong primitive models.
"""

import subprocess
from pathlib import Path


def model_files(root: Path) -> list[Path]:
    """Collect the VHDL and Verilog model files for simulation."""
    files = sorted((root / "rtl").glob("*.vhd"))
    files += sorted((root / "rtl").glob("*.v"))
    return files


def unisim_excludes(target: str) -> list[str]:
    """Return the unisim exclude list for a build target."""
    excludes = ["unisim/sequential", "unisim/simprim"]
    if target == "asic":
        excludes.append("unisim/gateway")
    return excludes


def launch(root: Path, target: str) -> None:
    """Launch iverilog over the model files with the exclude list."""
    cmd = ["iverilog"]
    for f in model_files(root):
        cmd += ["-s", f.stem]
    for excl in unisim_excludes(target):
        cmd += ["-y", excl]
    subprocess.run(cmd, check=True)
