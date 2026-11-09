"""Layout project edits, mutating in place. Ports are identified by `number`."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from EMShrimulator.core.errors import EditError
from EMShrimulator.core.layout.copper_geometry import check_piece
from EMShrimulator.core.layout.model import (
    Backdrill,
    CopperEdit,
    CopperObject,
    DifferentialPair,
    LayoutPort,
    LayoutPortKind,
    LayoutProject,
    Padstack,
    PadstackLayer,
    SingleEndedTrace,
    StackupLayer,
    Via,
)


def find_port(project: LayoutProject, number: int) -> Optional[LayoutPort]:
    return next((p for p in project.ports if p.number == number), None)


def copper_count(project: LayoutProject) -> int:
    return sum(1 for layer in project.stackup if layer.type == "copper")


def _next_number(project: LayoutProject) -> int:
    return max((p.number for p in project.ports), default=-1) + 1


def _default_plane(project: LayoutProject, layer: int) -> int:
    count = copper_count(project)
    if count < 2:
        return 0
    return layer + 1 if layer + 1 < count else layer - 1


def place_port(project: LayoutProject, x_mm: float, y_mm: float, layer: int = 0,
               snap: Optional[str] = None) -> int:
    """Drop a new microstrip port at a board-local position; returns its number."""
    number = _next_number(project)
    project.ports.append(LayoutPort(
        name=f"P{number + 1}", number=number, kind=LayoutPortKind.MICROSTRIP,
        x_mm=round(x_mm, 4), y_mm=round(y_mm, 4),
        layer=layer, plane=_default_plane(project, layer), reference="auto",
        excite=not project.ports, snap=snap,
    ))
    return number


def place_edge_port(project: LayoutProject, x1: float, y1: float, x2: float, y2: float,
                    rotation_deg: float, layer: int, kind: LayoutPortKind = LayoutPortKind.LUMPED,
                    snap: Optional[str] = None) -> int:
    """A port across a copper edge: centred on it, as wide as it, referenced to the nearest plane."""
    number = _next_number(project)
    width_um = math.hypot(x2 - x1, y2 - y1) * 1000.0
    project.ports.append(LayoutPort(
        name=f"P{number + 1}", number=number, kind=kind,
        x_mm=round((x1 + x2) / 2, 4), y_mm=round((y1 + y2) / 2, 4),
        rotation_deg=round(rotation_deg, 3) % 360.0, width_um=round(width_um, 2),
        layer=layer, plane=_default_plane(project, layer), reference="auto",
        excite=not project.ports, snap=snap,
    ))
    return number


def place_circuit_port(project: LayoutProject, pos: Tuple[float, float, int], neg: Tuple[float, float, int],
                       snap: Optional[str] = None, neg_snap: Optional[str] = None) -> int:
    """A circuit port from a + terminal to a - terminal, each (x_mm, y_mm, layer)."""
    number = _next_number(project)
    project.ports.append(LayoutPort(
        name=f"P{number + 1}", number=number, kind=LayoutPortKind.CIRCUIT,
        x_mm=round(pos[0], 4), y_mm=round(pos[1], 4), layer=pos[2],
        neg_x_mm=round(neg[0], 4), neg_y_mm=round(neg[1], 4), plane=neg[2],
        width_um=100.0, reference="layer", excite=not project.ports, snap=snap, neg_snap=neg_snap,
    ))
    return number


def move_port(project: LayoutProject, number: int, x_mm: float, y_mm: float,
              snap: Optional[str] = None) -> None:
    """Move a port. A circuit port moves as a whole, both terminals together."""
    port = find_port(project, number)
    if port is None:
        raise EditError("That port no longer exists.")
    dx, dy = x_mm - port.x_mm, y_mm - port.y_mm
    port.x_mm = round(x_mm, 4)
    port.y_mm = round(y_mm, 4)
    port.snap = snap
    if port.neg_x_mm is not None and port.neg_y_mm is not None:
        port.neg_x_mm = round(port.neg_x_mm + dx, 4)
        port.neg_y_mm = round(port.neg_y_mm + dy, 4)
        port.neg_snap = None


def move_port_terminal(project: LayoutProject, number: int, terminal: str, x_mm: float, y_mm: float,
                       layer: Optional[int] = None, snap: Optional[str] = None) -> None:
    """Move one terminal of a port ("pos" or "neg"), optionally onto another layer."""
    port = find_port(project, number)
    if port is None:
        raise EditError("That port no longer exists.")
    if terminal == "pos":
        port.x_mm, port.y_mm, port.snap = round(x_mm, 4), round(y_mm, 4), snap
        if layer is not None and layer >= 0:
            port.layer = layer
    elif terminal == "neg":
        port.neg_x_mm, port.neg_y_mm, port.neg_snap = round(x_mm, 4), round(y_mm, 4), snap
        if layer is not None and layer >= 0:
            port.plane = layer
    else:
        raise EditError(f"'{terminal}' is not a port terminal.")


def set_port_field(project: LayoutProject, number: int, field: str, value: Any) -> None:
    port = find_port(project, number)
    if port is None:
        raise EditError("That port no longer exists.")
    if field == "number":
        raise EditError("Use renumber_ports to change port numbers.")
    if field == "kind" and LayoutPortKind(value) == LayoutPortKind.CIRCUIT and port.neg_x_mm is None:
        # Start the - terminal on the reference layer straight under the + terminal.
        port.neg_x_mm, port.neg_y_mm = port.x_mm, port.y_mm
        port.reference = "layer"
    if field == "plane":
        port.reference = "layer"
    setattr(port, field, value)


def delete_port(project: LayoutProject, number: int) -> None:
    """Delete a port; its differential pair keeps the other port."""
    project.ports = [p for p in project.ports if p.number != number]
    for pair in project.diff_pairs:
        if pair.positive == number:
            pair.positive = None
        if pair.negative == number:
            pair.negative = None


def renumber_ports(project: LayoutProject) -> Dict[int, int]:
    """Renumber ports 0…N-1 in their current order; returns {old: new}."""
    ordered = sorted(project.ports, key=lambda p: p.number)
    mapping = {port.number: i for i, port in enumerate(ordered)}
    for port in ordered:
        port.number = mapping[port.number]
    for trace in project.traces:
        trace.start = mapping.get(trace.start, trace.start)
        trace.stop = mapping.get(trace.stop, trace.stop)
    for pair in project.diff_pairs:
        pair.positive = mapping.get(pair.positive, pair.positive)
        pair.negative = mapping.get(pair.negative, pair.negative)
    project.ports = ordered
    return mapping


# --- differential pairs ---

def pair_of(project: LayoutProject, number: int) -> Optional[Tuple[int, DifferentialPair]]:
    """The pair a port is in, with its index, or None."""
    return next(((i, pair) for i, pair in enumerate(project.diff_pairs) if number in (pair.positive, pair.negative)),
                None)


def _check_pair_port(project: LayoutProject, index: Optional[int], number: Optional[int], other: Optional[int]) -> None:
    if number is None:
        return
    if find_port(project, number) is None:
        raise EditError(f"There is no port P{number + 1}.")
    if number == other:
        raise EditError("A differential pair needs two different ports.")
    owner = pair_of(project, number)
    if owner is not None and owner[0] != index:
        raise EditError(f"P{number + 1} is already in differential pair {owner[1].name}: a port can be in one pair.")


def _next_pair_name(project: LayoutProject) -> str:
    taken = {pair.name for pair in project.diff_pairs}
    return next(f"D{i}" for i in range(1, len(taken) + 2) if f"D{i}" not in taken)


def add_diff_pair(project: LayoutProject, positive: Optional[int] = None, negative: Optional[int] = None) -> int:
    """A new pair of two ports (either may be left for later); returns its index.

    Mode impedances start at Z+ + Z− (differential) and their parallel (common).
    """
    _check_pair_port(project, None, positive, negative)
    _check_pair_port(project, None, negative, positive)
    ports = [find_port(project, n) for n in (positive, negative) if n is not None]
    z = [p.impedance for p in ports if p is not None] or [50.0]
    zp, zn = (z * 2)[:2]  # one port known: it stands for both
    project.diff_pairs.append(DifferentialPair(
        name=_next_pair_name(project), positive=positive, negative=negative,
        z_diff=round(zp + zn, 6), z_common=round(zp * zn / (zp + zn), 6)))
    return len(project.diff_pairs) - 1


def _pair(project: LayoutProject, index: int) -> DifferentialPair:
    if not 0 <= index < len(project.diff_pairs):
        raise EditError("That differential pair no longer exists.")
    return project.diff_pairs[index]


def set_diff_pair_field(project: LayoutProject, index: int, field: str, value: Any) -> None:
    pair = _pair(project, index)
    if field == "positive":
        _check_pair_port(project, index, value, pair.negative)
    elif field == "negative":
        _check_pair_port(project, index, value, pair.positive)
    elif field in ("z_diff", "z_common"):
        if value is None or float(value) <= 0:
            raise EditError("A reference impedance must be positive.")
        value = float(value)
    elif field == "name":
        value = str(value).strip()
        if not value:
            raise EditError("A differential pair needs a name.")
        if any(p.name == value for i, p in enumerate(project.diff_pairs) if i != index):
            raise EditError(f"There is already a differential pair named {value}.")
    else:
        raise EditError(f"'{field}' is not a differential pair setting.")
    setattr(pair, field, value)


def swap_diff_pair(project: LayoutProject, index: int) -> None:
    """Exchange the + and − ports: the pair's polarity."""
    pair = _pair(project, index)
    pair.positive, pair.negative = pair.negative, pair.positive


def delete_diff_pair(project: LayoutProject, index: int) -> None:
    _pair(project, index)
    del project.diff_pairs[index]


# --- stackup ---

def add_stackup_layer(project: LayoutProject, index: Optional[int] = None) -> int:
    layer = StackupLayer(name=f"Layer{len(project.stackup) + 1}", type="copper", thickness_mm=0.035)
    if index is None or index >= len(project.stackup):
        project.stackup.append(layer)
        return len(project.stackup) - 1
    project.stackup.insert(index, layer)
    return index


def set_stackup_field(project: LayoutProject, index: int, field: str, value: Any) -> None:
    if not 0 <= index < len(project.stackup):
        raise EditError("That stackup layer no longer exists.")
    setattr(project.stackup[index], field, value)


def delete_stackup_layer(project: LayoutProject, index: int) -> None:
    if 0 <= index < len(project.stackup):
        del project.stackup[index]


def move_stackup_layer(project: LayoutProject, index: int, delta: int) -> int:
    target = index + delta
    if not (0 <= index < len(project.stackup) and 0 <= target < len(project.stackup)):
        return index
    layers = project.stackup
    layers[index], layers[target] = layers[target], layers[index]
    return target


# --- nets under test ---

def add_trace(project: LayoutProject) -> None:
    project.traces.append(SingleEndedTrace(start=0, stop=1))


# --- copper ---

def copper_edits(project: LayoutProject, layer: Optional[str] = None) -> Dict[str, Dict[int, List[Dict]]]:
    """The copper edits as {layer: {object index: [pieces as dicts]}}, for one layer or all."""
    out: Dict[str, Dict[int, List[Dict]]] = {}
    for edit in project.copper_edits:
        if layer is None or edit.layer == layer:
            out.setdefault(edit.layer, {})[edit.index] = [obj.model_dump() for obj in edit.objects]
    return out


def set_copper(project: LayoutProject, layer: str, replacements: Dict[int, Optional[List[Dict]]]) -> None:
    """Replace imported objects of `layer` by index: pieces reshape or split, [] deletes, None restores."""
    for index, pieces in replacements.items():
        for piece in pieces or []:
            problem = check_piece(piece)
            if problem:
                raise EditError(problem)
    kept = [e for e in project.copper_edits if not (e.layer == layer and e.index in replacements)]
    for index, pieces in replacements.items():
        if pieces is not None:
            kept.append(CopperEdit(layer=layer, index=int(index), objects=[CopperObject(**p) for p in pieces]))
    project.copper_edits = sorted(kept, key=lambda e: (e.layer, e.index))


def revert_copper(project: LayoutProject, layer: Optional[str] = None) -> int:
    """Restore the imported copper on one layer or all; returns how many objects were edited."""
    before = len(project.copper_edits)
    project.copper_edits = [e for e in project.copper_edits if layer is not None and e.layer != layer]
    return before - len(project.copper_edits)


# --- padstacks and vias ---

def _padstack(project: LayoutProject, name: str) -> Padstack:
    ps = next((p for p in project.padstacks if p.name == name), None)
    if ps is None:
        raise EditError(f"There is no padstack named {name}.")
    return ps


def _vias(project: LayoutProject) -> List[Via]:
    if project.vias is None:
        project.vias = []
    return project.vias


def _via(project: LayoutProject, via_id: int) -> Via:
    via = next((v for v in _vias(project) if v.id == via_id), None)
    if via is None:
        raise EditError("That via no longer exists.")
    return via


def add_padstack(project: LayoutProject, like: Optional[str] = None) -> str:
    """A new padstack, a copy of `like` or a plain through via; returns its name."""
    base = _padstack(project, like).model_copy(deep=True) if like else Padstack(name="")
    stem = like or f"VIA_{base.hole_mm:g}_{base.pad_mm:g}"
    taken = {p.name for p in project.padstacks}
    base.name = next(n for n in [stem] + [f"{stem}_{i}" for i in range(2, len(taken) + 3)] if n not in taken)
    project.padstacks.append(base)
    return base.name


def set_padstack_field(project: LayoutProject, name: str, field: str, value: Any) -> None:
    ps = _padstack(project, name)
    if field == "name":
        value = str(value).strip()
        if not value:
            raise EditError("A padstack needs a name.")
        if value != name and any(p.name == value for p in project.padstacks):
            raise EditError(f"There is already a padstack named {value}.")
        for via in _vias(project):
            if via.padstack == name:
                via.padstack = value
    elif field in ("hole_mm", "plating_um", "fill_epsilon", "antipad_mm"):
        value = float(value)
        if value < 0 or (field in ("hole_mm", "fill_epsilon") and value <= 0):
            raise EditError("That has to be a positive number.")
    elif field == "pad_mm":
        value = None if value is None else float(value)
        if value is not None and value <= 0:
            raise EditError("A pad needs a diameter above zero; choose no pad instead.")
    elif field in ("start_layer", "stop_layer"):
        value = value or None
    elif field not in ("plated", "shape"):
        raise EditError(f"'{field}' is not a padstack setting.")
    setattr(ps, field, value)


def set_padstack_layer(project: LayoutProject, name: str, layer: str, pad_mm: Optional[float],
                       antipad_mm: float, same_as_default: bool = False) -> None:
    """One layer's pad (None: none) and antipad; `same_as_default` drops the layer's own row."""
    ps = _padstack(project, name)
    if pad_mm is not None and pad_mm <= 0:
        raise EditError("A pad needs a diameter above zero; choose no pad instead.")
    if antipad_mm < 0:
        raise EditError("An antipad cannot be negative.")
    rows = [r for r in ps.layers if r.layer != layer]
    if not same_as_default:
        rows.append(PadstackLayer(layer=layer, pad_mm=pad_mm, antipad_mm=antipad_mm))
    ps.layers = rows


def delete_padstack(project: LayoutProject, name: str) -> None:
    _padstack(project, name)
    used = sum(1 for v in _vias(project) if v.padstack == name)
    if used:
        raise EditError(f"{used} via(s) use {name}: give them another padstack first.")
    project.padstacks = [p for p in project.padstacks if p.name != name]


def add_via(project: LayoutProject, x_mm: float, y_mm: float, padstack: str, net: Optional[str] = None) -> int:
    """A new via of `padstack` at a board-local point; returns its id."""
    _padstack(project, padstack)
    vias = _vias(project)
    via_id = max((v.id for v in vias), default=0) + 1
    vias.append(Via(id=via_id, padstack=padstack, x_mm=round(x_mm, 6), y_mm=round(y_mm, 6), net=net))
    return via_id


def move_vias(project: LayoutProject, moves: Dict[int, Tuple[float, float]]) -> None:
    for via_id, (x, y) in moves.items():
        via = _via(project, int(via_id))
        via.x_mm, via.y_mm = round(float(x), 6), round(float(y), 6)


def delete_vias(project: LayoutProject, ids) -> None:
    """Remove vias, with the pads they took over: those are theirs."""
    gone = set(ids)
    project.vias = [v for v in _vias(project) if v.id not in gone]


def set_vias_field(project: LayoutProject, ids, field: str, value: Any) -> None:
    """One setting of several vias: "padstack", "net", "backdrill_top" or "backdrill_bottom"."""
    for via_id in ids:
        via = _via(project, int(via_id))
        if field == "padstack":
            _padstack(project, value)
            via.padstack = value
        elif field == "net":
            via.net = str(value).strip() or None if value is not None else None
        elif field in ("backdrill_top", "backdrill_bottom"):
            setattr(via, field, None if value is None else Backdrill(**value))
        else:
            raise EditError(f"'{field}' is not a via setting.")
    if field.startswith("backdrill") and value is not None:
        from EMShrimulator.core.layout.padstacks import layer_names, padstack_map, span

        layers, stacks = layer_names(project), padstack_map(project)
        index = {name: i for i, name in enumerate(layers)}
        for via_id in ids:
            via = _via(project, int(via_id))
            ps = stacks.get(via.padstack)
            start, stop = span(ps, layers) if ps is not None else (0, len(layers) - 1)
            k = index.get(value.get("keep_layer"))
            if k is None or not start <= k <= stop:
                raise EditError("A backdrill keeps a layer the via spans.")
            if ps is not None and float(value.get("diameter_mm") or 0) <= ps.hole_mm:
                raise EditError(f"A backdrill is wider than the hole it cuts ({ps.hole_mm * 1000:g} µm).")
            top, bottom = via.backdrill_top, via.backdrill_bottom
            if top is not None and bottom is not None and index.get(top.keep_layer, 0) > index.get(bottom.keep_layer, 0):
                raise EditError("The top and bottom backdrills would meet: the via would connect nothing.")


# --- settings ---

SETTING_ROOTS = {"frequency", "max_steps", "convergence", "pixel_size_um", "via", "export_field", "use_pml",
                 "description"}


def set_path(project: LayoutProject, path: str, value: Any) -> None:
    """Assign a board-wide setting by dotted path, e.g. `via.filling_epsilon`."""
    parts = path.split(".")
    if parts[0] not in SETTING_ROOTS:
        raise EditError(f"'{path}' is not a board setting.")
    node: Any = project
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)
