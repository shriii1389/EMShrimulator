"""Padstacks and vias. A padstack defines a drill, plating, layer span and per-layer pad and antipad; a via
is an instance of one, with optional backdrills. Via pads are written into the copper Gerbers.

Layers are named like the copper Gerbers ("F_Cu", "In1_Cu", "B_Cu") and ordered top to bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from EMShrimulator.core.layout.model import LayoutProject, Padstack, Via

Point = Tuple[float, float]
CIRCLE_SIDES = 32  # a pad drawn as a polygon, for previews


def layer_names(project: LayoutProject) -> List[str]:
    """The copper layers, top to bottom, with '.' as '_'."""
    return [layer.name.replace(".", "_") for layer in project.stackup if layer.type == "copper"]


def padstack_map(project: LayoutProject) -> Dict[str, Padstack]:
    return {ps.name: ps for ps in project.padstacks}


def span(ps: Padstack, layers: Sequence[str]) -> Tuple[int, int]:
    """The first and last layer index the padstack spans (a through via: all of them)."""
    if not layers:
        return 0, -1
    index = {name: i for i, name in enumerate(layers)}
    start = index.get(ps.start_layer, 0) if ps.start_layer else 0
    stop = index.get(ps.stop_layer, len(layers) - 1) if ps.stop_layer else len(layers) - 1
    return (start, stop) if start <= stop else (stop, start)


def drilled_span(via: Via, ps: Padstack, layers: Sequence[str]) -> Tuple[int, int]:
    """(first, last) index of the layers the via still connects."""
    start, stop = span(ps, layers)
    index = {name: i for i, name in enumerate(layers)}
    if via.backdrill_top is not None and via.backdrill_top.keep_layer in index:
        start = max(start, index[via.backdrill_top.keep_layer])
    if via.backdrill_bottom is not None and via.backdrill_bottom.keep_layer in index:
        stop = min(stop, index[via.backdrill_bottom.keep_layer])
    return start, stop


def layer_pad(ps: Padstack, layer: str) -> Tuple[Optional[float], float]:
    """(pad diameter or None, antipad diameter) on one layer."""
    for row in ps.layers:
        if row.layer == layer:
            return row.pad_mm, row.antipad_mm
    return ps.pad_mm, ps.antipad_mm


@dataclass
class LayerPad:
    """What a via is on one layer it spans."""

    layer: str
    pad_mm: Optional[float]  # disc diameter; None: the imported pad if `link` is set, else no pad
    link: Optional[int] = None  # the imported object standing as its pad
    link_offset: Point = (0.0, 0.0)
    antipad_mm: float = 0.0
    drill_mm: float = 0.0  # backdrill diameter on layers it passes; 0 elsewhere

    @property
    def has_pad(self) -> bool:
        return self.drill_mm <= 0 and (self.pad_mm is not None or self.link is not None)


def via_layers(via: Via, ps: Padstack, layers: Sequence[str]) -> List[LayerPad]:
    """The via on each layer of its padstack's span, top first."""
    start, stop = span(ps, layers)
    keep_from, keep_to = drilled_span(via, ps, layers)
    links = {link.layer.lower(): link for link in via.links}
    out = []
    for i in range(start, stop + 1):
        name = layers[i]
        pad, antipad = layer_pad(ps, name)
        link = links.get(name.lower())
        row = LayerPad(name, pad, antipad_mm=antipad)
        if ps.shape == "imported" and link is not None and _layer_wants_pad(ps, name):
            row.pad_mm, row.link, row.link_offset = None, link.index, (link.dx, link.dy)
        if i < keep_from and via.backdrill_top is not None:
            row.drill_mm = via.backdrill_top.diameter_mm
        elif i > keep_to and via.backdrill_bottom is not None:
            row.drill_mm = via.backdrill_bottom.diameter_mm
        out.append(row)
    return out


def owned_links(via: Via, layers: Sequence[str]) -> list:
    """The imported pads the via replaces, on layers the stackup has."""
    known = {name.lower() for name in layers}
    return [link for link in via.links if link.layer.lower() in known]


def _layer_wants_pad(ps: Padstack, layer: str) -> bool:
    """Whether an imported-shape padstack keeps a layer's imported pad."""
    row = next((r for r in ps.layers if r.layer == layer), None)
    return row is None or row.pad_mm is not None


def pad_radius(via: Via, ps: Optional[Padstack], layers: Sequence[str], visible: Optional[Iterable[str]] = None,
               imported: Optional[Dict[str, float]] = None) -> float:
    """The via's reach on the given layers: its largest pad's radius, or its hole's."""
    if ps is None:
        return 0.15
    shown = set(visible) if visible is not None else None
    r = ps.hole_mm / 2
    for row in via_layers(via, ps, layers):
        if shown is not None and row.layer not in shown or not row.has_pad:
            continue
        if row.pad_mm is not None:
            r = max(r, row.pad_mm / 2)
        elif imported is not None and row.layer in imported:
            r = max(r, imported[row.layer])
    return r


def circle(cx: float, cy: float, d: float, sides: int = CIRCLE_SIDES) -> List[Point]:
    r = d / 2
    return [(cx + r * math.cos(2 * math.pi * k / sides), cy + r * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]


# ------------------------------------------------------------------ the Gerbers

@dataclass
class LayerSpec:
    """What the vias put on one layer's Gerber: replaced object indices and flashes."""

    owned: List[int] = field(default_factory=list)
    flashes: List[Dict] = field(default_factory=list)


def layer_specs(project: LayoutProject) -> Dict[str, Dict]:
    """Per layer {"owned": [object indices], "flashes": [...]} for core.layout.copper. Empty without vias."""
    if not project.vias:
        return {}
    layers = layer_names(project)
    stacks = padstack_map(project)
    specs: Dict[str, LayerSpec] = {}
    clears: Dict[str, List[Dict]] = {}
    pads: Dict[str, List[Dict]] = {}
    for via in project.vias:
        for link in owned_links(via, layers):
            specs.setdefault(link.layer.lower(), LayerSpec()).owned.append(link.index)
        ps = stacks.get(via.padstack)
        if ps is None:
            continue
        for row in via_layers(via, ps, layers):
            key = row.layer.lower()  # one layer however the stackup and its Gerber spell it
            specs.setdefault(key, LayerSpec())
            if row.antipad_mm > 0:
                clears.setdefault(key, []).append(
                    {"x": via.x_mm, "y": via.y_mm, "d": row.antipad_mm, "dark": False, "net": None})
            if row.drill_mm > 0:
                clears.setdefault(key, []).append(
                    {"x": via.x_mm, "y": via.y_mm, "d": row.drill_mm, "dark": False, "net": None})
                continue
            if row.link is not None:
                dx, dy = row.link_offset
                pads.setdefault(key, []).append(
                    {"x": via.x_mm + dx, "y": via.y_mm + dy, "link": row.link, "dark": True, "net": via.net})
            elif row.pad_mm is not None and row.pad_mm > 0:
                pads.setdefault(key, []).append(
                    {"x": via.x_mm, "y": via.y_mm, "d": row.pad_mm, "dark": True, "net": via.net})
    out = {}
    for name, spec in specs.items():
        out[name] = {"owned": sorted(set(spec.owned)),
                     "flashes": clears.get(name, []) + pads.get(name, [])}
    return out


# ------------------------------------------------------------------ the solver

def sim_vias(project: LayoutProject) -> List[Dict]:
    """The plated vias and backdrilled holes for gerber2ems's vias.json."""
    layers = layer_names(project)
    stacks = padstack_map(project)
    out = []
    if not layers:
        return out
    for via in project.vias or []:
        ps = stacks.get(via.padstack)
        if ps is None or not ps.plated or ps.hole_mm <= 0:
            continue
        start, stop = span(ps, layers)
        top, bottom = drilled_span(via, ps, layers)
        if top > bottom:
            continue  # drilled through: no barrel left
        entry = {
            "x_mm": via.x_mm, "y_mm": via.y_mm, "hole_mm": ps.hole_mm, "plating_um": ps.plating_um,
            "fill_epsilon": ps.fill_epsilon, "top": top, "bottom": bottom,
            "top_stub_mm": None, "bottom_stub_mm": None, "backdrills": [],
        }
        if via.backdrill_top is not None and top > start:
            entry["top_stub_mm"] = max(via.backdrill_top.stub_mm, 0.0)
            entry["backdrills"].append({"side": "top", "diameter_mm": via.backdrill_top.diameter_mm})
        if via.backdrill_bottom is not None and bottom < stop:
            entry["bottom_stub_mm"] = max(via.backdrill_bottom.stub_mm, 0.0)
            entry["backdrills"].append({"side": "bottom", "diameter_mm": via.backdrill_bottom.diameter_mm})
        out.append(entry)
    return out


# ------------------------------------------------------------------ checks

def problems(project: LayoutProject) -> List[str]:
    """What is wrong with the padstacks and vias, in words; empty when nothing is."""
    out: List[str] = []
    layers = layer_names(project)
    stacks = padstack_map(project)
    missing = sorted({v.padstack for v in project.vias or [] if v.padstack not in stacks})
    if missing:
        out.append(f"Vias use padstacks that no longer exist: {', '.join(missing)}.")
    index = {name: i for i, name in enumerate(layers)}
    for ps in project.padstacks:
        for name in (ps.start_layer, ps.stop_layer):
            if name and name not in index:
                out.append(f"Padstack {ps.name} spans to {name}, which is not a copper layer of the stackup.")
    bad = 0
    for via in project.vias or []:
        ps = stacks.get(via.padstack)
        if ps is None:
            continue
        start, stop = span(ps, layers)
        for drill in (via.backdrill_top, via.backdrill_bottom):
            if drill is None:
                continue
            k = index.get(drill.keep_layer)
            if k is None or not start <= k <= stop or drill.diameter_mm <= ps.hole_mm:
                bad += 1
                break
    if bad:
        out.append(f"{bad} via(s) have a backdrill that keeps a layer outside the via's span, or is no wider "
                   "than the via's hole.")
    return out
