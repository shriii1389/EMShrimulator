"""Convert an ODB++ board into the same fab/ set a Gerber import produces: Gerber X2 copper with nets, an
Edge_Cuts outline, an Excellon drill file and stackup.json. Output starts at (0, 0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from EMShrimulator.core.layout.odbpp import (
    IDENTITY,
    Affine,
    Contour,
    Feature,
    LayerGeometry,
    ODBBoard,
    Stroke,
    Surface,
    load,
)

# Layer names gerber2ems keys off.
_KICAD_INNER = [f"In{i}_Cu" for i in range(1, 31)]
OUTLINE_WIDTH_MM = 0.05

ApertureKey = Tuple


def _fmt(v: float) -> str:
    """4.6 fixed-point, the format declared in the header."""
    return f"{int(round(v * 1e6)):d}"


def _num(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def _x2_escape(text: str) -> str:
    """An X2 attribute field: '%', '*', ',' and '\\' may not appear as themselves."""
    return "".join(f"\\u{ord(ch):04X}" if ch in "%*,\\" else ch for ch in text)


def _hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class GerberWriter:
    """One Gerber X2 file, written object by object in drawing order."""

    def __init__(self, dx: float, dy: float):
        self.dx, self.dy = dx, dy
        self.apertures: Dict[ApertureKey, int] = {}
        self.body: List[str] = []
        self.polarity = "D"
        self.net: Optional[str] = None
        self.aperture: Optional[int] = None
        self.mode = "G01"
        self.at: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------ state

    def _xy(self, x: float, y: float) -> str:
        return f"X{_fmt(x + self.dx)}Y{_fmt(y + self.dy)}"

    def polarity_dark(self, dark: bool) -> None:
        want = "D" if dark else "C"
        if want != self.polarity:
            self.body.append(f"%LP{want}*%")
            self.polarity = want

    def set_net(self, net: Optional[str]) -> None:
        if net == self.net:
            return
        self.body.append(f"%TO.N,{_x2_escape(net)}*%" if net else "%TD*%")
        self.net = net

    def _select(self, key: ApertureKey) -> None:
        code = self.apertures.setdefault(key, 10 + len(self.apertures))
        if code != self.aperture:
            self.body.append(f"D{code}*")
            self.aperture = code

    def _set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.body.append(f"{mode}*")
            self.mode = mode

    def _segment(self, s, x0: float, y0: float) -> None:
        if s[0] == "L":
            self._set_mode("G01")
            self.body.append(f"{self._xy(s[1], s[2])}D01*")
        else:
            self._set_mode("G02" if s[5] else "G03")
            self.body.append(f"{self._xy(s[1], s[2])}I{_fmt(s[3] - x0)}J{_fmt(s[4] - y0)}D01*")

    # ---------------------------------------------------------- objects

    def flash(self, key: ApertureKey, x: float, y: float) -> None:
        self._select(key)
        self.body.append(f"{self._xy(x, y)}D03*")
        self.at = (x, y)

    def stroke(self, width: float, x1: float, y1: float, x2: float, y2: float,
               arc: Optional[Tuple[float, float, bool]] = None) -> None:
        self._select(("C", round(width, 6)))
        if self.at is None or abs(self.at[0] - x1) > 1e-9 or abs(self.at[1] - y1) > 1e-9:
            self.body.append(f"{self._xy(x1, y1)}D02*")
        self._segment(("L", x2, y2) if arc is None else ("A", x2, y2, arc[0], arc[1], arc[2]), x1, y1)
        self.at = (x2, y2)

    def contour_strokes(self, width: float, contour: Contour) -> None:
        x0, y0 = contour.start
        for s in contour.segments:
            self.stroke(width, x0, y0, s[1], s[2], None if s[0] == "L" else (s[3], s[4], s[5]))
            x0, y0 = s[1], s[2]

    def region(self, contours: Sequence[Contour]) -> None:
        """Each island with its holes cut in, as one region."""
        groups: List[Tuple[Contour, List[Contour]]] = []
        for contour in contours:
            if contour.hole:
                if groups:
                    groups[-1][1].append(contour)
            else:
                groups.append((contour, []))
        for island, holes in groups:
            if island.signed_area() < 0:
                island = island.reversed()
            self.body.append("G36*")
            self.body.append(f"{self._xy(*island.start)}D02*")
            x0, y0 = island.start
            for s in island.segments:
                self._segment(s, x0, y0)
                x0, y0 = s[1], s[2]
            for hole in holes:
                if hole.signed_area() > 0:
                    hole = hole.reversed()
                self._set_mode("G01")
                self.body.append(f"{self._xy(*hole.start)}D01*")
                x0, y0 = hole.start
                for s in hole.segments:
                    self._segment(s, x0, y0)
                    x0, y0 = s[1], s[2]
                self._set_mode("G01")
                self.body.append(f"{self._xy(*island.start)}D01*")
            self.body.append("G37*")
        self.at = None

    # ------------------------------------------------------------ text

    def text(self, function: str, comment: str) -> str:
        head = [
            "%TF.GenerationSoftware,EMShrimulator,ODB++ import*%",
            f"%TF.FileFunction,{function}*%",
            "%TF.FilePolarity,Positive*%",
            "%FSLAX46Y46*%",
            "%MOMM*%",
            f"G04 {comment}*",
        ]
        for key, code in self.apertures.items():
            if key[0] == "C":
                params = _num(key[1]) + (f"X{_num(key[2])}" if len(key) > 2 else "")
            else:
                params = f"{_num(key[1])}X{_num(key[2])}"
            head.append(f"%ADD{code}{key[0]},{params}*%")
        head += ["G75*", "%LPD*%", "G01*"]
        return "\n".join(head + self.body + ["M02*"]) + "\n"


def _emit(w: GerberWriter, feature: Feature, t: Affine = IDENTITY, invert: bool = False,
          net: Optional[str] = None) -> None:
    """One feature, placed by `t` (a user symbol's pads are features about its origin)."""
    positive = feature.positive != invert
    w.set_net(net if net is not None else feature.net)
    if isinstance(feature, Surface):
        w.polarity_dark(positive)
        w.region([c.transformed(t) for c in feature.contours] if t is not IDENTITY else feature.contours)
        return
    symbol = feature.symbol
    if isinstance(feature, Stroke):
        x1, y1 = t.apply(feature.x1, feature.y1)
        x2, y2 = t.apply(feature.x2, feature.y2)
        arc = None
        if feature.arc is not None:
            xc, yc = t.apply(feature.arc[0], feature.arc[1])
            arc = (xc, yc, feature.arc[2] != t.flips)
        width = symbol.stroke_width
        if width <= 0 or symbol.kind in ("none", "unknown"):
            return
        w.polarity_dark(positive)
        if symbol.kind == "rect" and arc is None:
            # A square symbol swept along a line: hull of its corners at both ends.
            hw, hh = symbol.width / 2, symbol.height / 2
            corners = [(x + sx * hw, y + sy * hh) for x, y in ((x1, y1), (x2, y2)) for sx in (-1, 1) for sy in (-1, 1)]
            hull = _hull(corners)
            w.region([Contour(hull[0], [("L", x, y) for x, y in hull[1:] + hull[:1]])])
        else:
            w.stroke(width, x1, y1, x2, y2, arc)
        return
    # A pad.
    placed = t * feature.placement
    kind = symbol.kind
    if kind in ("none", "unknown"):
        return
    if kind == "user":
        for sub in symbol.features:
            _emit(w, sub, placed, not positive, net if net is not None else feature.net)
        w.set_net(net if net is not None else feature.net)
        return
    w.polarity_dark(positive)
    x, y = placed.tx, placed.ty
    turns = placed.quarter_turns
    if kind == "round" and symbol.width > 0:
        w.flash(("C", round(symbol.width, 6)), x, y)
    elif kind == "donut" and symbol.width > symbol.inner > 0:
        w.flash(("C", round(symbol.width, 6), round(symbol.inner, 6)), x, y)
    elif kind in ("rect", "oval") and turns is not None and symbol.width > 0 and symbol.height > 0:
        width, height = (symbol.height, symbol.width) if turns else (symbol.width, symbol.height)
        w.flash(("R" if kind == "rect" else "O", round(width, 6), round(height, 6)), x, y)
    else:
        outline = symbol.outline()
        if outline:
            w.region([c.transformed(placed) for c in outline])


def write_copper(geom: LayerGeometry, path: Path, dx: float, dy: float, function: str,
                 negative_outline: Optional[Sequence[Contour]] = None) -> None:
    """A copper layer. A negative layer is the outline filled, then every feature in the opposite polarity."""
    w = GerberWriter(dx, dy)
    invert = negative_outline is not None
    if negative_outline:
        w.polarity_dark(True)
        w.region(negative_outline)
    for feature in geom.features:
        _emit(w, feature, invert=invert)
    path.write_text(w.text(function, f"Converted by EMShrimulator Studio from ODB++ layer {geom.name}"),
                    encoding="utf-8")


def write_outline(contours: Sequence[Contour], path: Path, dx: float, dy: float) -> None:
    w = GerberWriter(dx, dy)
    for contour in contours:
        w.contour_strokes(OUTLINE_WIDTH_MM, contour)
    path.write_text(w.text("Profile,NP", "Board outline, from the ODB++ step profile"), encoding="utf-8")


def write_excellon(holes: Sequence[Tuple[float, float, float]], path: Path, dx: float, dy: float) -> None:
    """Metric Excellon with explicit decimals, one tool per diameter."""
    by_diameter: Dict[float, List[Tuple[float, float]]] = {}
    for x, y, d in holes:
        by_diameter.setdefault(round(d, 4), []).append((x, y))

    lines = ["M48", "; EMShrimulator Studio - converted from ODB++", "METRIC,TZ"]
    tools = sorted(by_diameter)
    for i, d in enumerate(tools, start=1):
        lines.append(f"T{i}C{d:.3f}")
    lines.append("%")
    lines.append("G90")
    lines.append("G05")
    for i, d in enumerate(tools, start=1):
        lines.append(f"T{i}")
        for x, y in by_diameter[d]:
            lines.append(f"X{x + dx:.4f}Y{y + dy:.4f}")
    lines.append("T0")
    lines.append("M30")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kicad_layer_name(index: int, total: int) -> str:
    if index == 0:
        return "F_Cu"
    if index == total - 1:
        return "B_Cu"
    return _KICAD_INNER[min(index - 1, len(_KICAD_INNER) - 1)]


def _file_function(index: int, total: int) -> str:
    side = "Top" if index == 0 else "Bot" if index == total - 1 else "Inr"
    return f"Copper,L{index + 1},{side}"


def _outline_contours(board: ODBBoard) -> List[Contour]:
    contours: List[Contour] = []
    for feature in board.outline.features if board.outline else []:
        if isinstance(feature, Surface):
            contours.extend(feature.contours)
    return contours


def convert(board: ODBBoard, fab_dir: Path, board_name: str) -> Dict:
    """Write fab/ files for an ODB++ board; returns an import report."""
    fab_dir.mkdir(parents=True, exist_ok=True)
    outline = _outline_contours(board)
    if outline:
        xs = [x for c in outline for x, _y in c.samples()]
        ys = [y for c in outline for _x, y in c.samples()]
        (lo_x, lo_y), (hi_x, hi_y) = (min(xs), min(ys)), (max(xs), max(ys))
    else:
        (lo_x, lo_y), (hi_x, hi_y) = board.bounds_mm()
        # No profile: the copper's bounding rectangle stands in.
        outline = [Contour((lo_x, lo_y), [("L", hi_x, lo_y), ("L", hi_x, hi_y), ("L", lo_x, hi_y), ("L", lo_x, lo_y)])]
    # The outline is stroked, so its box is half a stroke wider.
    dx, dy = OUTLINE_WIDTH_MM / 2 - lo_x, OUTLINE_WIDTH_MM / 2 - lo_y
    width, height = hi_x - lo_x + OUTLINE_WIDTH_MM, hi_y - lo_y + OUTLINE_WIDTH_MM

    warnings: List[str] = []
    written: List[str] = []
    copper_names: List[str] = []
    total = len(board.copper)
    unknown, approximate, negative = set(), set(), []
    for i, geom in enumerate(board.copper):
        kicad_name = _kicad_layer_name(i, total)
        out = fab_dir / f"{board_name}-{kicad_name}.gbr"
        layer = board.copper_layers[i] if i < len(board.copper_layers) else None
        is_negative = layer is not None and layer.polarity == "NEGATIVE"
        if is_negative:
            negative.append(geom.name)
        write_copper(geom, out, dx, dy, _file_function(i, total), outline if is_negative else None)
        written.append(out.name)
        copper_names.append(kicad_name)
        unknown |= geom.unknown_symbols
        approximate |= geom.approximate_symbols

    edge = fab_dir / f"{board_name}-Edge_Cuts.gbr"
    write_outline(outline, edge, dx, dy)
    written.append(edge.name)

    drill = fab_dir / f"{board_name}-PTH.drl"
    write_excellon(board.drills, drill, dx, dy)

    if len(board.steps) > 1:
        warnings.append(f"The product has {len(board.steps)} steps ({', '.join(board.steps)}); "
                        f"'{board.step}' was imported.")
    if unknown:
        names = sorted(unknown)
        warnings.append(f"{len(names)} pad symbol(s) were not understood and are left out: "
                        f"{', '.join(names[:8])}{'…' if len(names) > 8 else ''}.")
    if approximate:
        warnings.append(f"Thermal reliefs ({', '.join(sorted(approximate)[:4])}) are drawn as solid rings.")
    if negative:
        warnings.append(f"Negative layer(s) {', '.join(negative)} were converted to positive copper inside the "
                        "outline.")
    through = {board.copper_layers[0].name.lower(), board.copper_layers[-1].name.lower()} \
        if board.copper_layers else set()
    spans = [layer.name for layer in board.layers if layer.type == "DRILL" and layer.start_name and layer.end_name
             and {layer.start_name.lower(), layer.end_name.lower()} != through]
    if spans:
        warnings.append(f"Blind or buried drill layers ({', '.join(spans)}) are modelled as through holes: "
                        "gerber2ems reads a single drill file.")
    nets = board.net_names
    if not nets:
        warnings.append("The product has no net data (steps/…/eda/data), so the board cannot be coloured by net.")

    return {
        "step": board.step,
        "steps": list(board.steps),
        "copper_layers": copper_names,
        "odb_copper_layers": [geom.name for geom in board.copper],
        "gerber_files": written,
        "drill_file": drill.name,
        "vias": len(board.drills),
        "nets": len(nets),
        "width_mm": width,
        "height_mm": height,
        "warnings": warnings,
        "matrix_layers": [
            {"name": layer.name, "type": layer.type, "side": layer.side, "context": layer.context}
            for layer in board.layers
        ],
    }


def stackup_from_odb(board: ODBBoard, default_thickness_mm: float = 0.2,
                     copper_thickness_mm: float = 0.035,
                     epsilon: float = 4.3, loss_tangent: float = 0.02) -> List[Dict]:
    """Stackup skeleton from the ODB++ layer order. Dielectric properties are placeholders."""
    layers: List[Dict] = []
    total = len(board.copper)
    for i in range(total):
        layers.append({
            "name": _kicad_layer_name(i, total),
            "type": "copper",
            "thickness_mm": copper_thickness_mm,
            "material": None,
            "epsilon": None,
            "loss_tangent": None,
        })
        if i < total - 1:
            layers.append({
                "name": f"Dielectric {i + 1}",
                "type": "core" if i % 2 else "prepreg",
                "thickness_mm": default_thickness_mm,
                "material": "FR4 (placeholder — set from your fab's datasheet)",
                "epsilon": epsilon,
                "loss_tangent": loss_tangent,
            })
    return layers


def import_archive(archive: Path, work_dir: Path, fab_dir: Path, board_name: str,
                   step: str | None = None) -> Dict:
    board = load(archive, work_dir, step=step)
    if not board.copper:
        raise ValueError("the product has no copper layers on the board (no SIGNAL or POWER_GROUND layer in "
                         "its matrix)")
    report = convert(board, fab_dir, board_name)
    report["stackup"] = stackup_from_odb(board)
    (fab_dir / "odbpp-import.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
