"""Excellon drill reading via gerbonara, shifted into the board-local frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from gerbonara import ExcellonFile

BBox = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class Hole:
    """One drilled hole, in board-local millimetres from the outline corner."""

    x_mm: float
    y_mm: float
    diameter_mm: float
    plated: bool

    def to_dict(self) -> dict:
        return {
            "x_mm": self.x_mm, "y_mm": self.y_mm,
            "diameter_mm": self.diameter_mm, "plated": self.plated,
        }


def find_drill_file(fab_dir: Path, plated_only: bool = True) -> Optional[Path]:
    """The drill file to use: the plated one when there is a choice."""
    candidates = sorted(p for p in fab_dir.glob("*.drl") if p.is_file())
    if not candidates:
        return None
    if plated_only:
        plated = [p for p in candidates if "NPTH" not in p.name.upper()]
        if plated:
            return plated[0]
    return candidates[0]


def read_raw(path: Path) -> List[Tuple[float, float, float, bool]]:
    """Every hole in the file as (x, y, diameter, plated), in file millimetres."""
    excellon = ExcellonFile.open(str(path))
    out: List[Tuple[float, float, float, bool]] = []
    for obj in excellon.objects:
        # Slots are skipped; only round holes become vias.
        x = getattr(obj, "x", None)
        y = getattr(obj, "y", None)
        aperture = getattr(obj, "aperture", None)
        if x is None or y is None or aperture is None:
            continue
        unit = getattr(obj, "unit", None)
        to_mm = 25.4 if str(unit) == "LengthUnit inch" or getattr(unit, "name", "") == "inch" else 1.0
        diameter = getattr(aperture, "diameter", 0.0) or 0.0
        ap_unit = getattr(aperture, "unit", None)
        ap_to_mm = 25.4 if getattr(ap_unit, "name", "") == "inch" else 1.0
        plated = getattr(obj, "plated", None)
        out.append((x * to_mm, y * to_mm, diameter * ap_to_mm, plated is not False))
    return out


def holes(drill_path: Path, board_bbox: Optional[BBox]) -> List[Hole]:
    """Holes in board-local mm. `board_bbox` is the Edge_Cuts bbox, or None."""
    raw = read_raw(drill_path)
    if not raw or board_bbox is None:
        return [Hole(x, y, d, p) for x, y, d, p in raw]

    (min_x, min_y), (max_x, max_y) = board_bbox
    # Slack for holes on the outline edge.
    pad = 1.0
    inside = sum(
        1 for x, y, _d, _p in raw
        if min_x - pad <= x <= max_x + pad and min_y - pad <= y <= max_y + pad
    )

    if inside >= max(1, len(raw) // 2):
        # Holes land on the board as absolute coordinates, so shift them.
        return [Hole(x - min_x, y - min_y, d, p) for x, y, d, p in raw]

    # Otherwise the exporter already wrote them relative to a board origin.
    return [Hole(x, y, d, p) for x, y, d, p in raw]


def holes_for_project(fab_dir: Path, board_bbox: Optional[BBox]) -> List[Hole]:
    """Convenience wrapper: find the drill file and read it, or return none."""
    path = find_drill_file(fab_dir)
    if path is None:
        return []
    try:
        return holes(path, board_bbox)
    except Exception:  # noqa: BLE001 - an unreadable drill file is not fatal
        return []
