"""Layer images and the mm <-> pixel mapping for the 2D board canvas.

Board coordinates are mm from the Edge_Cuts bbox with Y up; image rows run down.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from EMShrimulator.core.layout import drill
from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm, render_layer_preview_png, render_layer_png
from EMShrimulator.core.layout.gerber_shapes import INDEX_COLUMNS, layer_shapes

# KiCad copper-layer file naming: <board>-F_Cu.gbr / -In1_Cu.gbr / -B_Cu.gbr
_COPPER_RE = re.compile(r"-(F_Cu|B_Cu|In\d+_Cu)\.gbr$", re.IGNORECASE)
_EDGE_RE = re.compile(r"Edge_Cuts\.gbr$", re.IGNORECASE)

_LAYER_COLORS = {
    "F_Cu": "#d4423c",
    "B_Cu": "#3c7cd4",
}
_INNER_COLORS = ["#4aa564", "#c9a227", "#9a6fd4", "#2fa5a5", "#d4772f", "#8fb03e"]


@dataclass
class CopperLayer:
    name: str        # KiCad layer name, e.g. "F_Cu"
    file: str        # gerber filename inside fab/
    index: int       # position among the copper gerbers, top first
    color: str
    # Index among the stackup's copper layers; None if not in the stackup.
    sim_index: Optional[int] = None


def _layer_sort_key(name: str) -> Tuple[int, int]:
    upper = name.upper()
    if upper == "F_CU":
        return (0, 0)
    if upper == "B_CU":
        return (2, 0)
    m = re.match(r"IN(\d+)_CU", upper)
    return (1, int(m.group(1)) if m else 0)


def stackup_metal_names(stackup: Optional[List[Dict]]) -> List[str]:
    """Stackup copper layer names in order, with '.' replaced by '_' (F.Cu -> F_Cu)."""
    if not stackup:
        return []
    return [str(layer.get("name", "")).replace(".", "_") for layer in stackup if layer.get("type") == "copper"]


def copper_layer_name(filename: str) -> Optional[str]:
    """The copper layer a Gerber's file name names ("F_Cu"), or None."""
    m = _COPPER_RE.search(filename)
    return m.group(1) if m else None


def copper_layers(fab_dir: Path, stackup: Optional[List[Dict]] = None) -> List[CopperLayer]:
    found: List[Tuple[str, str]] = []
    for path in sorted(fab_dir.glob("*_Cu.gbr")):
        m = _COPPER_RE.search(path.name)
        if m:
            found.append((m.group(1), path.name))
    found.sort(key=lambda pair: _layer_sort_key(pair[0]))

    metals = stackup_metal_names(stackup)
    sim_index_by_name = {name.lower(): i for i, name in enumerate(metals)}

    out: List[CopperLayer] = []
    inner = 0
    for i, (name, filename) in enumerate(found):
        color = _LAYER_COLORS.get(name.upper().replace("_CU", "_Cu"))
        if color is None:
            color = _LAYER_COLORS.get(name) or _INNER_COLORS[inner % len(_INNER_COLORS)]
            if name.upper().startswith("IN"):
                inner += 1
        out.append(CopperLayer(
            name=name, file=filename, index=i, color=color,
            sim_index=sim_index_by_name.get(name.lower()),
        ))
    return out


def edge_cuts_file(fab_dir: Path) -> Optional[Path]:
    for path in sorted(fab_dir.glob("*.gbr")):
        if _EDGE_RE.search(path.name):
            return path
    return None


def _layer_dict(layer: CopperLayer) -> Dict:
    return {
        "name": layer.name, "file": layer.file, "index": layer.index,
        "color": layer.color, "sim_index": layer.sim_index,
    }


# Preview resolution targets a pixel count.
PREVIEW_TARGET_PX = 2000      # long edge, before clamping
PREVIEW_MAX_PIXELS = 6_000_000  # total, so a big board cannot blow up memory
PREVIEW_MIN_PX_PER_MM = 4.0
PREVIEW_MAX_PX_PER_MM = 240.0
KEEP_RENDERS_PER_LAYER = 4


def preview_resolution(width_mm: float, height_mm: float) -> float:
    """Pixels per mm for the canvas raster of a board this size."""
    longest = max(width_mm, height_mm, 1e-6)
    px_per_mm = PREVIEW_TARGET_PX / longest
    px_per_mm = max(PREVIEW_MIN_PX_PER_MM, min(PREVIEW_MAX_PX_PER_MM, px_per_mm))
    area_px = width_mm * height_mm * px_per_mm * px_per_mm
    if area_px > PREVIEW_MAX_PIXELS:
        px_per_mm *= (PREVIEW_MAX_PIXELS / area_px) ** 0.5
    return px_per_mm


def board_info(project_dir: Path, stackup: Optional[List[Dict]] = None,
               preview_px_per_mm: Optional[float] = None) -> Dict:
    """Board extent, copper layer list, and the mm<->pixel mapping for the canvas."""
    fab_dir = project_dir / "fab"
    layers = copper_layers(fab_dir, stackup) if fab_dir.exists() else []
    edge = edge_cuts_file(fab_dir) if fab_dir.exists() else None

    if edge is None:
        return {
            "ready": False,
            "reason": "No Edge_Cuts gerber found. Upload the board outline (…-Edge_Cuts.gbr) "
                      "along with the copper layers — it defines the coordinate origin everything "
                      "else is measured from.",
            "layers": [_layer_dict(layer) for layer in layers],
        }

    (min_x, min_y), (max_x, max_y) = edge_cuts_bbox_mm(edge)
    width_mm = max_x - min_x
    height_mm = max_y - min_y

    px_per_mm = preview_px_per_mm or preview_resolution(width_mm, height_mm)
    px_w = max(int(round(width_mm * px_per_mm)), 1)
    px_h = max(int(round(height_mm * px_per_mm)), 1)

    return {
        "ready": True,
        "board": {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "origin_mm": [min_x, min_y],
        },
        "preview": {
            "px_per_mm": px_per_mm,
            "width_px": px_w,
            "height_px": px_h,
            # y_mm = height_mm - py / px_per_mm
            "y_axis": "up",
        },
        "layers": [_layer_dict(layer) for layer in layers],
        "edge_cuts": edge.name,
    }


def _preview_source(project_dir: Path, layer_name: str) -> Tuple[Path, Path]:
    """(gerber to draw, Edge_Cuts gerber that frames it) for a canvas layer."""
    fab_dir = project_dir / "fab"
    edge = edge_cuts_file(fab_dir)
    if edge is None:
        raise FileNotFoundError("no Edge_Cuts gerber in this project")
    if layer_name.lower() in ("edge_cuts", "outline"):
        return edge, edge
    match = next((layer for layer in copper_layers(fab_dir) if layer.name.lower() == layer_name.lower()), None)
    if match is None:
        raise FileNotFoundError(f"no copper layer named '{layer_name}'")
    return fab_dir / match.file, edge


def _source_tag(project_dir: Path, layer_name: str, variant: str) -> str:
    """Cache key from the layer and outline file stats plus what was made from them."""
    source, edge = _preview_source(project_dir, layer_name)
    parts = [variant]
    for path in (source, edge):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def preview_tag(project_dir: Path, layer_name: str, px_per_mm: float) -> str:
    """Cache key from the layer and outline file stats plus resolution."""
    return _source_tag(project_dir, layer_name, f"{px_per_mm:.4f}")


def render_preview(project_dir: Path, layer_name: str, px_per_mm: float) -> bytes:
    """Transparent PNG of one layer framed to the Edge_Cuts bbox, cached under the project."""
    source, edge = _preview_source(project_dir, layer_name)
    tag = preview_tag(project_dir, layer_name, px_per_mm)
    cache = project_dir / "cache" / "preview" / f"{layer_name}-{tag}.png"
    if cache.is_file():
        try:
            return cache.read_bytes()
        except FileNotFoundError:
            pass  # removed by another worker meanwhile: render again

    bounds = edge_cuts_bbox_mm(edge)
    png = render_layer_preview_png(source, bounds, 1000.0 / px_per_mm)

    cache.parent.mkdir(parents=True, exist_ok=True)
    stale = sorted(
        (q for q in cache.parent.glob(f"{layer_name}-*.png") if q != cache),
        key=lambda q: q.stat().st_mtime, reverse=True,
    )
    for old in stale[KEEP_RENDERS_PER_LAYER - 1:]:
        old.unlink(missing_ok=True)
    tmp = cache.with_suffix(".png.part")
    tmp.write_bytes(png)
    tmp.replace(cache)
    return png


def sync_copper(project_dir: Path, edits: Dict[str, Dict[int, List[Dict]]],
                vias: Optional[Dict[str, Dict]] = None) -> List[str]:
    """Write each copper layer's Gerber with its edits and via pads; returns the layers whose file changed."""
    fab_dir = project_dir / "fab"
    edge = edge_cuts_file(fab_dir) if fab_dir.is_dir() else None
    if edge is None:
        return []
    from EMShrimulator.core.layout import copper

    bounds = edge_cuts_bbox_mm(edge)
    specs = {name.lower(): spec for name, spec in (vias or {}).items()}
    changed = []
    for layer in copper_layers(fab_dir):
        layer_edits = dict(edits.get(layer.name, {}))
        spec = specs.get(layer.name.lower()) or {}
        for index in spec.get("owned", ()):
            layer_edits[int(index)] = []  # the via writes its own pad
        if copper.sync_layer(fab_dir, layer.file, bounds, layer_edits, spec.get("flashes", ())):
            changed.append(layer.name)
    return changed


def copper_pieces(project_dir: Path, layer_name: str) -> Dict:
    """One layer's imported copper as editable pieces: {"pieces", "editable", "reason"}."""
    from EMShrimulator.core.layout import copper

    fab_dir = project_dir / "fab"
    source, edge = _preview_source(project_dir, layer_name)
    return copper.layer_pieces(copper.original_path(fab_dir, source.name), edge_cuts_bbox_mm(edge))


def canvas_data(project_dir: Path, stackup: Optional[List[Dict]] = None,
                edits: Optional[Dict[str, Dict[int, List[Dict]]]] = None,
                vias: Optional[Dict[str, Dict]] = None) -> Tuple[Dict, List[Dict]]:
    """`board_info` (with nets) and `drill_holes` in one call, after writing the edited copper."""
    if edits is not None:
        sync_copper(project_dir, edits, vias)
    info = board_info(project_dir, stackup)
    if info.get("ready"):
        info["nets"] = board_nets(project_dir)
    return info, (drill_holes(project_dir) if info.get("ready") else [])


_NET_ATTRIBUTE = re.compile(rb"%TO\.N,([^*%]*)\*%")


def board_nets(project_dir: Path) -> List[Dict]:
    """Every net named in the copper Gerbers, sorted, with its layers: [{"name", "layers"}, ...]."""
    fab_dir = project_dir / "fab"
    on: Dict[str, List[str]] = {}
    for layer in copper_layers(fab_dir) if fab_dir.is_dir() else []:
        try:
            data = (fab_dir / layer.file).read_bytes()
        except OSError:
            continue
        for raw in set(_NET_ATTRIBUTE.findall(data)):
            name = raw.split(b",", 1)[0].decode("utf-8", errors="replace")
            if name:
                on.setdefault(name, []).append(layer.name)
    return [{"name": name, "layers": on[name]} for name in sorted(on, key=lambda n: (n.lower(), n))]


def render_preview_alpha(project_dir: Path, layer_name: str, px_per_mm: float) -> Tuple[int, int, bytes]:
    """(width, height, 8-bit coverage) of one canvas layer. Runs in a worker process."""
    import io

    from PIL import Image

    png = render_preview(project_dir, layer_name, px_per_mm)
    with Image.open(io.BytesIO(png)) as image:
        alpha = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
        return alpha.width, alpha.height, alpha.tobytes()


SHAPES_VERSION = 2  # bump when gerber_shapes' output changes


def preview_shapes(project_dir: Path, layer_name: str) -> Dict[str, np.ndarray]:
    """One canvas layer as vector shapes, cached next to its PNGs as plain arrays (no pickle)."""
    source, edge = _preview_source(project_dir, layer_name)
    tag = _source_tag(project_dir, layer_name, f"shapes-v{SHAPES_VERSION}")
    cache = project_dir / "cache" / "preview" / f"{layer_name}-shapes-{tag}.npz"
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as stored:
                index, blob, nets = stored["index"], stored["blob"], stored["nets"]
            if index.ndim == 2 and index.shape[1] == INDEX_COLUMNS and blob.dtype == np.uint8 and nets.ndim == 1:
                return {"index": index, "blob": blob, "nets": nets}
        except (OSError, ValueError, KeyError):
            pass  # unreadable: build it again

    shapes = layer_shapes(source, edge_cuts_bbox_mm(edge))
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Two workers may build the same layer at once; losing the race is fine.
    tmp = cache.with_suffix(f".{os.getpid()}.part")
    try:
        for old in cache.parent.glob(f"{layer_name}-shapes-*.npz"):
            if old != cache:
                old.unlink(missing_ok=True)
        with open(tmp, "wb") as fh:
            np.savez(fh, index=shapes["index"], blob=shapes["blob"], nets=shapes["nets"])
        tmp.replace(cache)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return shapes


def drill_holes(project_dir: Path) -> List[Dict]:
    """Via/hole positions in board-local mm."""
    fab_dir = project_dir / "fab"
    if not fab_dir.is_dir():
        return []
    edge = edge_cuts_file(fab_dir)
    bbox = edge_cuts_bbox_mm(edge) if edge is not None else None
    return [hole.to_dict() for hole in drill.holes_for_project(fab_dir, bbox)]


def ensure_simulation_pngs(project_dir: Path, pixel_size_um: float) -> List[str]:
    """Render the black/white PNGs gerber2ems meshes from."""
    fab_dir = project_dir / "fab"
    edge = edge_cuts_file(fab_dir)
    if edge is None:
        raise FileNotFoundError("no Edge_Cuts gerber in this project")
    bounds = edge_cuts_bbox_mm(edge)
    out_dir = project_dir / "ems" / "geometry"
    written = []
    for layer in copper_layers(fab_dir):
        out = out_dir / (layer.file.rpartition("-")[2].replace(".gbr", ".png"))
        render_layer_png(fab_dir / layer.file, bounds, out, pixel_size_um)
        written.append(out.name)
    return written


def data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
