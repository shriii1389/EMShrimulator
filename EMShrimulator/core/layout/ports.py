"""Port references: which copper layer a port's return current flows on.

"auto" takes the nearest layer with a plane (not just a trace) around the port, from the canvas rasters.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from EMShrimulator.core.layout import board
from EMShrimulator.core.layout.model import LayoutPort, LayoutPortKind, LayoutProject

# Fraction of the area around the port that must be copper to count as a plane.
PLANE_COVERAGE = 0.6
# Minimum half-size of the area checked.
MIN_PROBE_HALF_MM = 0.5

_MASKS: Dict[Tuple[str, str], np.ndarray] = {}
_MASK_LIMIT = 32


def metal_z_mm(project: LayoutProject) -> List[float]:
    """Height of each stackup copper layer, top at 0 going negative."""
    z, out = 0.0, []
    for layer in project.stackup:
        if layer.type == "copper":
            out.append(z)
        elif layer.type in ("core", "prepreg"):
            z -= layer.thickness_mm or 0.0
    return out


def metal_names(project: LayoutProject) -> List[str]:
    return [layer.name.replace(".", "_") for layer in project.stackup if layer.type == "copper"]


@dataclass
class ReferenceChoice:
    plane: Optional[int]
    coverage: float  # of the chosen layer around the port, 0..1
    found: bool  # False when no layer is a plane and the nearest was taken
    note: str

    def to_dict(self) -> Dict:
        return {"plane": self.plane, "coverage": self.coverage, "found": self.found, "note": self.note}


class CopperMasks:
    """Per-layer copper masks for one board, from the canvas preview rasters."""

    def __init__(self, project_dir: Path, project: LayoutProject):
        self.project_dir = project_dir
        stackup = [layer.model_dump() for layer in project.stackup]
        info = board.board_info(project_dir, stackup)
        self.ready = bool(info.get("ready"))
        self.layer_names: Dict[int, str] = {}
        if not self.ready:
            return
        self.ppm = float(info["preview"]["px_per_mm"])
        self.width_mm = float(info["board"]["width_mm"])
        self.height_mm = float(info["board"]["height_mm"])
        for layer in info["layers"]:
            if layer.get("sim_index") is not None:
                self.layer_names[int(layer["sim_index"])] = layer["name"]

    def mask(self, index: int) -> Optional[np.ndarray]:
        name = self.layer_names.get(index)
        if not self.ready or name is None:
            return None
        try:
            tag = board.preview_tag(self.project_dir, name, self.ppm)
        except FileNotFoundError:
            return None
        key = (str(self.project_dir), f"{name}-{tag}")
        cached = _MASKS.get(key)
        if cached is None:
            png = board.render_preview(self.project_dir, name, self.ppm)
            image = Image.open(BytesIO(png)).convert("RGBA")
            cached = np.asarray(image)[:, :, 3] > 127
            if len(_MASKS) >= _MASK_LIMIT:
                _MASKS.pop(next(iter(_MASKS)))
            _MASKS[key] = cached
        return cached

    def coverage(self, index: int, x0: float, y0: float, x1: float, y1: float) -> Optional[float]:
        """Copper fraction of layer `index` in a board-local mm box, clipped to the board."""
        mask = self.mask(index)
        if mask is None:
            return None
        x0, x1 = max(min(x0, x1), 0.0), min(max(x0, x1), self.width_mm)
        y0, y1 = max(min(y0, y1), 0.0), min(max(y0, y1), self.height_mm)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        h, w = mask.shape
        c0, c1 = int(x0 * self.ppm), int(np.ceil(x1 * self.ppm))
        r0, r1 = int((self.height_mm - y1) * self.ppm), int(np.ceil((self.height_mm - y0) * self.ppm))
        c0, c1 = max(0, min(c0, w - 1)), max(1, min(c1, w))
        r0, r1 = max(0, min(r0, h - 1)), max(1, min(r1, h))
        window = mask[r0:r1, max(c0, 0):max(c1, c0 + 1)]
        return float(window.mean()) if window.size else 0.0


def nearest_reference(project: LayoutProject, port: LayoutPort, masks: CopperMasks) -> ReferenceChoice:
    """The layer an "auto" port references: nearest in z with a plane under the port."""
    zs = metal_z_mm(project)
    names = metal_names(project)
    if port.layer >= len(zs) or len(zs) < 2:
        return ReferenceChoice(None, 0.0, False, "the stackup has no second copper layer to reference")
    z0 = zs[port.layer]
    # Nearest first; at equal distance the layer below wins.
    candidates = sorted((i for i in range(len(zs)) if i != port.layer),
                        key=lambda i: (round(abs(zs[i] - z0), 9), zs[i] > z0))
    half = max(2.0 * port.width_um / 1000.0, MIN_PROBE_HALF_MM)
    box = (port.x_mm - half, port.y_mm - half, port.x_mm + half, port.y_mm + half)

    measured: List[Tuple[int, float]] = []
    for index in candidates:
        cov = masks.coverage(index, *box)
        if cov is None:
            continue
        measured.append((index, cov))
        if cov >= PLANE_COVERAGE:
            return ReferenceChoice(index, cov, True, f"{names[index]} ({cov:.0%} copper under the port)")
    fallback = candidates[0]
    best = max(measured, key=lambda m: m[1], default=(fallback, 0.0))
    return ReferenceChoice(
        fallback, dict(measured).get(fallback, 0.0), False,
        f"no plane under the port (best was {names[best[0]]} at {best[1]:.0%}); using the nearest layer, "
        f"{names[fallback]}")


def resolve_references(project: LayoutProject, project_dir: Path) -> Dict[int, ReferenceChoice]:
    """The resolved reference of every "auto" port, by port number."""
    auto = [p for p in project.ports if p.reference == "auto" and p.kind != LayoutPortKind.CIRCUIT]
    if not auto:
        return {}
    masks = CopperMasks(project_dir, project)
    if not masks.ready:
        return {p.number: ReferenceChoice(None, 0.0, False, "the board has not been imported yet") for p in auto}
    return {p.number: nearest_reference(project, p, masks) for p in auto}


def resolved_project(project: LayoutProject, project_dir: Path) -> Tuple[LayoutProject, Dict[int, ReferenceChoice]]:
    """A copy with every "auto" plane filled in: what gerber2ems is actually handed."""
    choices = resolve_references(project, project_dir)
    out = project.model_copy(deep=True)
    for port in out.ports:
        choice = choices.get(port.number)
        if choice is not None and choice.plane is not None:
            port.plane = choice.plane
    return out, choices
