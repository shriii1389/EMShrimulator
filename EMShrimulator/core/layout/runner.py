"""Build the gerber2ems command and environment for a layout project directory."""
from __future__ import annotations

from typing import Dict, List

from EMShrimulator.core import workers
from EMShrimulator.core.layout.model import LayoutProject

STEP_FLAGS = {"geometry": "-g", "simulate": "-s", "postprocess": "-p", "all": "-a"}


def build_command(steps: List[str], project: LayoutProject | None = None, debug: bool = True) -> List[str]:
    """steps: any combination of 'geometry', 'simulate', 'postprocess', or ['all']."""
    unknown = [s for s in steps if s not in STEP_FLAGS]
    if unknown:
        raise ValueError(f"unknown step(s): {', '.join(unknown)}")
    args = [STEP_FLAGS[s] for s in steps]
    if debug:
        args.append("-d")
    if project and project.export_field:
        args.extend(["--export-field", *project.export_field])
    return workers.command("gerber2ems", *args)


def build_env() -> Dict[str, str]:
    return workers.environment()
