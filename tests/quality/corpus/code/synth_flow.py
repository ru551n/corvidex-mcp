"""Synthesis flow: create the project, run the synthesis script, and
collect the bitstream.

The flow is driven by a Tcl script; the Python wrapper only assembles
the command line and checks the result.
"""

import subprocess
from pathlib import Path


def create_project(root: Path, name: str) -> Path:
    """Create the project directory layout for a synthesis run."""
    project = root / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "tcl").mkdir(exist_ok=True)
    (project / "reports").mkdir(exist_ok=True)
    return project


def run_synthesis(project: Path, top: str) -> Path:
    """Run the synthesis Tcl script and return the bitstream path."""
    tcl = project / "tcl" / "synth.tcl"
    subprocess.run(["vivado", "-mode", "batch", "-source", str(tcl), "-tclargs", top], check=True)
    bitstream = project / "reports" / f"{top}.bit"
    if not bitstream.exists():
        raise RuntimeError(f"synthesis finished but no bitstream at {bitstream}")
    return bitstream