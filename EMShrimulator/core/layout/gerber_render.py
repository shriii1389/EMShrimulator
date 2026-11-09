"""Gerber -> PNG with gerbonara + resvg, framed to the Edge_Cuts bbox.

render_layer_png: white copper on black, for gerber2ems. render_layer_preview_png: white on transparent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import resvg_py
from gerbonara import GerberFile
from gerbonara.aperture_macros import primitive as _macro_primitive

# Silence a leftover debug print in gerbonara's polygon aperture macro.
_macro_primitive.print = lambda *_args, **_kwargs: None

BBox = Tuple[Tuple[float, float], Tuple[float, float]]

_BBOXES: Dict[Tuple[str, int, int], BBox] = {}


def edge_cuts_bbox_mm(edge_cuts_path: Path) -> BBox:
    """The outline's bounding box, parsed once per version of the file."""
    stat = Path(edge_cuts_path).stat()
    key = (str(edge_cuts_path), stat.st_mtime_ns, stat.st_size)
    bbox = _BBOXES.get(key)
    if bbox is None:
        (x0, y0), (x1, y1) = GerberFile.open(str(edge_cuts_path)).bounding_box(unit="mm")
        bbox = ((float(x0), float(y0)), (float(x1), float(y1)))
        if len(_BBOXES) >= 64:
            _BBOXES.clear()
        _BBOXES[key] = bbox
    return bbox


def _rasterize(svg: str, pixel_size_um: float) -> bytes:
    dpi = 25400.0 / pixel_size_um
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, dpi=dpi))


def render_layer_png(gerber_path: Path, bounds_mm: BBox, out_png_path: Path, pixel_size_um: float) -> None:
    layer = GerberFile.open(str(gerber_path))
    svg = layer.to_svg(fg="white", bg="black", force_bounds=bounds_mm)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    out_png_path.write_bytes(_rasterize(str(svg), pixel_size_um))


def render_layer_preview_png(gerber_path: Path, bounds_mm: BBox, pixel_size_um: float) -> bytes:
    """Copper in white on transparent, tinted per layer by the canvas."""
    layer = GerberFile.open(str(gerber_path))
    svg = layer.to_svg(fg="white", bg="none", force_bounds=bounds_mm)
    return _rasterize(str(svg), pixel_size_um)
