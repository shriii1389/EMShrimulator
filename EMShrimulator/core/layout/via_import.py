"""Read the drill file into padstacks and vias, once, when a board is imported.

Holes are registered onto the copper's pad flashes (scale and offset that land the most holes on pads), then
each hole takes over the pads centred on it. Alike holes share a padstack.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from EMShrimulator.core.layout import drill
from EMShrimulator.core.layout.board import copper_layers, edge_cuts_file
from EMShrimulator.core.layout.copper import original_path, read_layer
from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm

SCALES = (1.0, 10.0, 0.1, 100.0, 0.01, 25.4, 1 / 25.4)
VOTE_BIN_MM = 0.01
MATCH_MM = 0.02  # a registered hole this near a pad centre is on it
FALLBACK_MATCH_FRACTION = 0.5  # fraction of the hole diameter a pad may be off when nothing registered
SAMPLE = 48


def _pad_flashes(project_dir: Path, bounds) -> Dict[str, List[Tuple[int, float, float, Optional[float], Optional[str]]]]:
    """Per copper layer, its flashes as imported: (object index, x, y, circle diameter or None, net)."""
    from gerbonara.apertures import CircleAperture
    from gerbonara.utils import MM

    fab_dir = project_dir / "fab"
    out = {}
    for layer in copper_layers(fab_dir):
        try:
            data = read_layer(original_path(fab_dir, layer.file), bounds)
        except Exception:  # noqa: BLE001 - an unreadable layer just links no pads
            continue
        rows = []
        for index, (piece, obj) in enumerate(zip(data.pieces, data.objects)):
            if piece.kind != "pad" or not piece.points or not piece.dark or not piece.editable:
                continue
            aperture = getattr(obj, "aperture", None)
            d = None
            if isinstance(aperture, CircleAperture) and not getattr(aperture, "hole_dia", None):
                try:
                    d = float(aperture.equivalent_width(MM))
                except Exception:  # noqa: BLE001
                    d = None
            x, y = piece.points[0]
            rows.append((index, x, y, d, piece.net))
        out[layer.name] = rows
    return out


def _nearest(points: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each point, the index of the nearest target and the distance to it."""
    best = np.zeros(len(points), int)
    dist = np.full(len(points), np.inf)
    if not len(targets):
        return best, dist
    for start in range(0, len(points), 256):
        chunk = points[start:start + 256]
        d = np.hypot(chunk[:, None, 0] - targets[None, :, 0], chunk[:, None, 1] - targets[None, :, 1])
        best[start:start + 256] = d.argmin(axis=1)
        dist[start:start + 256] = d.min(axis=1)
    return best, dist


def register(raw: np.ndarray, pads: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """(scale, dx, dy, fraction matched) from drill mm to board-local mm, or None if no transform fits."""
    if len(raw) == 0 or len(pads) == 0:
        return None
    step = max(1, len(raw) // SAMPLE)
    sample = raw[::step][:SAMPLE]
    need = min(3, len(raw))
    best = None
    for scale in SCALES:
        offsets = (pads[None, :, :] - scale * sample[:, None, :]).reshape(-1, 2)
        bins, counts = np.unique(np.round(offsets / VOTE_BIN_MM).astype(np.int64), axis=0, return_counts=True)
        top = int(counts.argmax())
        if counts[top] < need:
            continue
        offset = bins[top] * VOTE_BIN_MM
        for _ in range(2):  # refine on what it lands: the median residual
            moved = scale * raw + offset
            near, dist = _nearest(moved, pads)
            hit = dist <= MATCH_MM * 2
            if not hit.any():
                break
            offset = offset + np.median(pads[near[hit]] - moved[hit], axis=0)
        moved = scale * raw + offset
        _near, dist = _nearest(moved, pads)
        fraction = float(np.mean(dist <= MATCH_MM))
        if best is None or fraction > best[3] + 1e-9:
            best = (scale, float(offset[0]), float(offset[1]), fraction)
    if best is None or best[3] < 0.5:
        return None
    return best


def _drill_files(fab_dir: Path) -> List[Path]:
    return sorted(p for p in fab_dir.glob("*.drl") if p.is_file())


def extract(project_dir: Path, plating_um: float = 25.0, fill_epsilon: float = 1.0) -> Dict:
    """{"padstacks", "vias", "note"} found in the project's fab/."""
    fab_dir = project_dir / "fab"
    edge = edge_cuts_file(fab_dir) if fab_dir.is_dir() else None
    if edge is None:
        return {"padstacks": [], "vias": [], "note": "no outline"}
    bounds = edge_cuts_bbox_mm(edge)
    raw: List[Tuple[float, float, float, bool]] = []
    for path in _drill_files(fab_dir):
        try:
            holes = drill.read_raw(path)
        except Exception:  # noqa: BLE001 - an unreadable drill file adds no holes
            continue
        unplated = "NPTH" in path.name.upper()
        raw += [(x, y, d, p and not unplated) for x, y, d, p in holes]
    if not raw:
        return {"padstacks": [], "vias": [], "note": "no holes"}

    flashes = _pad_flashes(project_dir, bounds)
    all_pads = np.array([(x, y) for rows in flashes.values() for _i, x, y, _d, _n in rows], float).reshape(-1, 2)
    xy = np.array([(x, y) for x, y, _d, _p in raw], float)
    fit = register(xy, all_pads)
    if fit is not None:
        scale, dx, dy, fraction = fit
        local = xy * scale + (dx, dy)
        # A wrong unit scales tools and coordinates; a wrong number format only the coordinates.
        unit_error = abs(scale - 25.4) < 1e-9 or abs(scale - 1 / 25.4) < 1e-9
        diameters = [d * scale if unit_error else d for _x, _y, d, _p in raw]
        note = f"registered on the pads ({fraction:.0%} of holes on one)"
        if scale != 1.0:
            note += f", its coordinates read at {scale:g}× the file's"
        tolerance = [MATCH_MM] * len(raw)
    else:
        legacy = drill.holes(_drill_files(fab_dir)[0], bounds) if len(_drill_files(fab_dir)) == 1 else None
        if legacy is not None and len(legacy) == len(raw):
            local = np.array([(h.x_mm, h.y_mm) for h in legacy], float)
        else:
            (x0, y0), _ = bounds
            local = xy - (x0, y0) if _mostly_inside(xy, bounds) else xy
        diameters = [d for _x, _y, d, _p in raw]
        note = "placed by the drill file's own frame (no pads to register it on)"
        tolerance = [max(d * FALLBACK_MATCH_FRACTION, MATCH_MM) for d in diameters]

    # Each hole takes the nearest unclaimed pad flash on each layer.
    links: List[Dict[str, Tuple[int, float, float, Optional[float], Optional[str]]]] = [dict() for _ in raw]
    for name, rows in flashes.items():
        if not rows:
            continue
        targets = np.array([(x, y) for _i, x, y, _d, _n in rows], float)
        near, dist = _nearest(local, targets)
        claimed = set()
        for h in np.argsort(dist):
            k = int(near[h])
            if dist[h] <= tolerance[h] and k not in claimed:
                claimed.add(k)
                links[int(h)][name] = rows[k]

    layer_order = [layer.name for layer in copper_layers(fab_dir)]
    stacks: Dict[tuple, Dict] = {}
    vias = []
    for h, ((_x, _y, _d, plated), (lx, ly), d) in enumerate(zip(raw, local, diameters)):
        found = links[h]
        circles = [row for row in found.values() if row[3] is not None]
        if circles:  # centred on its pads, whose coordinates are finer
            lx = sum(row[1] for row in circles) / len(circles)
            ly = sum(row[2] for row in circles) / len(circles)
        imported = any(row[3] is None for row in found.values())
        signature = tuple(None if name not in found else
                          ("imported" if imported else round(found[name][3], 4)) for name in layer_order)
        key = (bool(plated), round(d, 4), imported, signature)
        if key not in stacks:
            stacks[key] = _padstack(key, layer_order, plating_um, fill_epsilon)
        nets = Counter(row[4] for row in found.values() if row[4])
        vias.append({
            "id": h + 1, "padstack": key, "x_mm": round(float(lx), 6), "y_mm": round(float(ly), 6),
            "net": nets.most_common(1)[0][0] if nets else None,
            "links": [{"layer": name, "index": row[0], "dx": round(row[1] - float(lx), 6),
                       "dy": round(row[2] - float(ly), 6)} for name, row in found.items()],
        })
    _name_padstacks(list(stacks.values()))
    for via in vias:
        via["padstack"] = stacks[via["padstack"]]["name"]
    padstacks = sorted(stacks.values(), key=lambda ps: (not ps["plated"], ps["hole_mm"], ps["name"]))
    return {"padstacks": padstacks, "vias": vias, "note": note}


def _mostly_inside(xy: np.ndarray, bounds) -> bool:
    (x0, y0), (x1, y1) = bounds
    inside = (xy[:, 0] >= x0 - 1) & (xy[:, 0] <= x1 + 1) & (xy[:, 1] >= y0 - 1) & (xy[:, 1] <= y1 + 1)
    return int(inside.sum()) >= max(1, len(xy) // 2)


def _padstack(key: tuple, layer_order: Sequence[str], plating_um: float, fill_epsilon: float) -> Dict:
    plated, hole, imported, signature = key
    pads = [s for s in signature if s is not None]
    if imported:
        common = None
    else:
        common = Counter(pads).most_common(1)[0][0] if pads else None
    rows = []
    for name, sig in zip(layer_order, signature):
        if imported:
            if sig is None:
                rows.append({"layer": name, "pad_mm": None, "antipad_mm": 0.0})
        elif sig != common:
            rows.append({"layer": name, "pad_mm": sig, "antipad_mm": 0.0})
    return {
        "name": "", "hole_mm": hole, "plated": plated, "plating_um": plating_um,
        "fill_epsilon": fill_epsilon, "shape": "imported" if imported else "circle",
        # New vias of an imported-shape padstack get its own pad.
        "pad_mm": common if not imported else round(hole + 0.5, 4), "antipad_mm": 0.0, "layers": rows,
    }


def _name_padstacks(stacks: List[Dict]) -> None:
    """Padstack names like VIA_0.2_0.45, PTH_1.0 or NPTH_3.2; repeats get a letter."""
    taken: Dict[str, int] = {}
    for ps in stacks:
        hole = f"{ps['hole_mm']:g}"
        if not ps["plated"]:
            base = f"NPTH_{hole}"
        elif ps["shape"] == "imported":
            base = f"PTH_{hole}"
        elif ps["pad_mm"] is None:
            base = f"VIA_{hole}_nopad"
        else:
            base = f"VIA_{hole}_{ps['pad_mm']:g}"
        n = taken.get(base, 0)
        taken[base] = n + 1
        ps["name"] = base if n == 0 else f"{base}_{chr(ord('a') + n - 1) if n <= 26 else n}"


def apply(project, found: Dict) -> None:
    """Install what `extract` found on a project, dropping copper edits of pads the vias took over."""
    from EMShrimulator.core.layout.model import Padstack, Via

    project.padstacks = [Padstack(**ps) for ps in found.get("padstacks", [])]
    project.vias = [Via(**via) for via in found.get("vias", [])]
    owned = {(link.layer, link.index) for via in project.vias for link in via.links}
    project.copper_edits = [e for e in project.copper_edits if (e.layer, e.index) not in owned]
