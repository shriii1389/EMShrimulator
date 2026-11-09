"""Editable copper: the imported Gerbers' traces, pours and pads as objects that can be reshaped.

The imported file is never touched: `sync` writes fab/<file> from the original in fab/.original/, with each
edited object's statements replaced and everything else kept byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from gerbonara import graphic_objects as go
from gerbonara import graphic_primitives as gp
from gerbonara.apertures import CircleAperture
from gerbonara.rs274x import GerberFile, GerberParser
from gerbonara.utils import MM, InterpMode

from EMShrimulator.core.layout.copper_geometry import (
    PAD,
    REGION,
    TRACE,
    Piece,
    Point,
    arc_center,
    check_piece,
    height_of,
)
from EMShrimulator.core.layout.gerber_render import BBox
from EMShrimulator.core.layout.gerber_shapes import _StampedObjects, _arc, _arc_poly, _rect, net_of

ORIGINAL_DIR = ".original"  # inside fab/: the imported Gerbers, as imported

# ------------------------------------------------------------------ reading a Gerber with its statements

_COMMAND = re.compile(r"G04.*?\*\s*|%.*?%\s*|[^*%]*\*\s*", re.DOTALL)  # gerbonara's own split
_DRAW = re.compile(r"D0*([123])$")
_APERTURE_DEF = re.compile(r"%ADD(\d+)")
_MODE = {InterpMode.LINEAR: "G01", InterpMode.CIRCULAR_CW: "G02", InterpMode.CIRCULAR_CCW: "G03"}


@dataclass
class _Record:
    """Where an object's statements are in the file, and the drawing state after them."""

    first: int  # command indices, inclusive
    last: int
    point: Optional[Tuple[float, float]]  # file units
    mode: InterpMode
    quadrant_single: Optional[bool]  # G74 in force
    aperture: Optional[int]  # D code


class _Parser(GerberParser):
    """gerbonara's parser, recording which commands made each object."""

    def __init__(self, target):
        super().__init__(target)
        self.spans: List[Tuple[int, int]] = []
        self.records: List[Optional[_Record]] = []
        self.first_draw: Optional[int] = None  # offset of the first drawing command; new apertures go before it
        self._first: Dict[int, int] = {}
        self._last: Dict[int, _Record] = {}
        self._alive: List[object] = []  # keeps regions whose ids are keys above

    def _state(self, first: int, last: int) -> _Record:
        gs = self.graphics_state
        return _Record(first, last, gs.point, gs.interpolation_mode, self.multi_quadrant_mode,
                       getattr(gs.aperture, "original_number", None))

    def _split_commands(self, data):
        self.lineno = 1
        objects = self.target.objects
        for match in _COMMAND.finditer(data):
            raw = match[0]
            cmd = raw.strip().strip("%").rstrip("*")
            if cmd:
                index = len(self.spans)
                self.spans.append((match.start(), match.end()))
                self.line = cmd
                region, size, seen = self.current_region, len(self.current_region or ()), len(objects)
                yield cmd  # resumed once the parser has acted on it
                self._after(index, cmd, region, size, seen)
            self.lineno += raw.count("\n")
        self.lineno = 0
        self.line = ""

    def _after(self, index: int, cmd: str, region_before, size_before: int, seen: int) -> None:
        draw = _DRAW.search(cmd) is not None
        if self.first_draw is None and (draw or cmd.startswith("G36")):
            self.first_draw = self.spans[index][0]
        region = self.current_region
        if region is not None and draw:
            key = id(region)
            if key not in self._first:
                self._first[key] = index
                self._alive.append(region)
            if len(region) > (size_before if region is region_before else 0):
                self._last[key] = self._state(self._first[key], index)
        for obj in self.target.objects[seen:]:
            if cmd.startswith("SR"):
                self.records.append(None)  # a step-and-repeat copy: no statements of its own
            elif isinstance(obj, go.Region):
                self.records.append(self._last.get(id(obj)))
            else:
                self.records.append(self._state(index, index))


@dataclass
class LayerFile:
    """A copper Gerber read for editing: text, objects, statement spans and pieces."""

    text: str
    objects: List[object]
    records: List[Optional[_Record]]
    spans: List[Tuple[int, int]]
    first_draw: Optional[int]
    settings: object
    origin: Point
    pieces: List[Piece]
    editable: bool
    reason: str = ""


def _mm(value: float, unit) -> float:
    return MM(value, unit)


def _piece(obj, origin: Point, editable: bool) -> Optional[Piece]:
    ox, oy = origin
    unit = obj.unit
    net, dark = net_of(obj), bool(obj.polarity_dark)

    def local(x, y) -> Point:
        return _mm(x, unit) - ox, _mm(y, unit) - oy

    if isinstance(obj, (go.Line, go.Arc)):
        a, b = local(obj.x1, obj.y1), local(obj.x2, obj.y2)
        try:
            width = float(obj.aperture.equivalent_width(MM)) if obj.aperture is not None else 0.0
        except Exception:  # noqa: BLE001 - an exotic aperture: drawn, but not given a width
            width = 0.0
        height = 0.0
        if isinstance(obj, go.Arc):
            cx, cy = obj.center
            height = height_of(a, b, local(cx, cy), obj.clockwise)
        full_circle = isinstance(obj, go.Arc) and math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-9
        return Piece(TRACE, [a, b], [height], width, net, dark, editable and not full_circle and width > 0)
    if isinstance(obj, go.Region):
        outline = [local(x, y) for x, y in obj.outline]
        centers = list(obj.arc_centers) + [None] * len(outline)
        heights = []
        for i in range(len(outline) - 1):
            arc = centers[i]
            if arc and arc[0] is not None:
                cx, cy = arc[1]
                heights.append(height_of(outline[i], outline[i + 1], local(cx, cy), bool(arc[0])))
            else:
                heights.append(0.0)
        if len(outline) > 1 and math.hypot(outline[-1][0] - outline[0][0], outline[-1][1] - outline[0][1]) < 1e-9:
            outline = outline[:-1]  # closed back to its start: the last edge ends on vertex 0
        else:
            heights.append(0.0)  # open: closed by a straight edge
        if len(outline) < 3:
            return None
        return Piece(REGION, outline, heights[:len(outline)], 0.0, net, dark, editable)
    if isinstance(obj, go.Flash):
        cx, cy = local(obj.x, obj.y)
        return Piece(PAD, [(cx, cy)], [], 0.0, net, dark, editable, _pad_outline(obj, (cx + ox, cy + oy)))
    return None


def _pad_outline(obj, centre_mm: Point) -> List[List[Point]]:
    """A flash's shape as polylines about its centre, for drawing and picking it."""
    out: List[List[Point]] = []
    try:
        prims = list(obj.to_primitives(unit=MM))
    except Exception:  # noqa: BLE001 - an exotic macro: a small square stands in
        prims = []
    cx, cy = centre_mm
    for prim in prims:
        if isinstance(prim, gp.Circle):
            start, segments = (prim.x + prim.r, prim.y), _arc(prim.x, prim.y, prim.x + prim.r, prim.y,
                                                              prim.x + prim.r, prim.y, False)
        elif isinstance(prim, gp.Rectangle):
            start, segments = _rect(prim)
        elif isinstance(prim, gp.ArcPoly) and len(prim.outline):
            start, segments = _arc_poly(prim)
        elif isinstance(prim, gp.Line):
            r = prim.width / 2
            start, segments = (prim.x1, prim.y1), [((prim.x2, prim.y2),)]
            out.append([(x - cx, y - cy) for x, y in ((prim.x1 - r, prim.y1 - r), (prim.x1 + r, prim.y1 + r),
                                                     (prim.x2 + r, prim.y2 + r), (prim.x2 - r, prim.y2 - r))])
            continue
        else:
            continue
        poly = [start]
        for seg in segments:
            if len(seg) == 1:
                poly.append(seg[0])
            else:  # a cubic: its points along the way
                p0 = poly[-1]
                c1, c2, p3 = seg
                for k in (0.25, 0.5, 0.75, 1.0):
                    u = 1 - k
                    poly.append((u ** 3 * p0[0] + 3 * u * u * k * c1[0] + 3 * u * k * k * c2[0] + k ** 3 * p3[0],
                                 u ** 3 * p0[1] + 3 * u * u * k * c1[1] + 3 * u * k * k * c2[1] + k ** 3 * p3[1]))
        out.append([(x - cx, y - cy) for x, y in poly])
    if not out:
        s = 0.1
        out.append([(-s, -s), (s, -s), (s, s), (-s, s)])
    return out


_READ_CACHE: Dict[tuple, LayerFile] = {}
_READ_CACHE_SIZE = 4


def read_layer(path: Path, bounds_mm: BBox) -> LayerFile:
    """A copper Gerber parsed for `patch`, cached since dense ones take seconds."""
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, tuple(map(tuple, bounds_mm)))
    if key in _READ_CACHE:
        return _READ_CACHE[key]
    layer = _read_layer(path, bounds_mm)
    while len(_READ_CACHE) >= _READ_CACHE_SIZE:
        _READ_CACHE.pop(next(iter(_READ_CACHE)))
    _READ_CACHE[key] = layer
    return layer


def _read_layer(path: Path, bounds_mm: BBox) -> LayerFile:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    gerber = GerberFile()
    parser = _Parser(gerber)
    gerber.objects = _StampedObjects(parser)
    parser.parse(text, filename=str(path))
    gs = parser.graphics_state
    reason = ""
    if parser.file_settings.is_incremental:
        reason = "it uses incremental coordinates"
    elif gs.image_offset != (0, 0) or gs.image_rotation or gs.image_mirror != (False, False) \
            or tuple(gs.image_scale) != (1.0, 1.0):
        reason = "it uses the deprecated image offset, rotation, mirroring or scaling"
    editable = not reason
    origin = (float(bounds_mm[0][0]), float(bounds_mm[0][1]))
    pieces = []
    for obj, record in zip(gerber.objects, parser.records):
        piece = _piece(obj, origin, editable and record is not None)
        if piece is None:  # kept in place, empty, so indices match the file's objects
            piece = Piece(PAD, [], editable=False)
        pieces.append(piece)
    return LayerFile(text, list(gerber.objects), parser.records, parser.spans, parser.first_draw,
                     parser.file_settings, origin, pieces, editable, reason)


def layer_pieces(path: Path, bounds_mm: BBox) -> Dict:
    """A layer's imported pieces and whether it can be edited."""
    layer = read_layer(path, bounds_mm)
    return {"pieces": layer.pieces, "editable": layer.editable, "reason": layer.reason}


# ------------------------------------------------------------------ writing

class _Writer:
    def __init__(self, layer: LayerFile):
        self.layer = layer
        self.settings = layer.settings
        numbers = [int(n) for n in _APERTURE_DEF.findall(layer.text)]
        self.next_aperture = max(numbers, default=9) + 1
        self.new_apertures: Dict[float, int] = {}  # width in file units -> D code

    def num(self, value_mm: float) -> str:
        return self.settings.write_gerber_value(self.settings.unit(value_mm, MM))

    def xy(self, p: Point) -> str:
        ox, oy = self.layer.origin
        return f"X{self.num(p[0] + ox)}Y{self.num(p[1] + oy)}"

    def aperture_for(self, width_mm: float) -> int:
        width = round(self.settings.unit(width_mm, MM), 6)
        if width not in self.new_apertures:
            self.new_apertures[width] = self.next_aperture
            self.next_aperture += 1
        return self.new_apertures[width]

    def edge(self, a: Point, b: Point, height: float, out: List[str], mode: List[str]) -> None:
        center = arc_center(a, b, height)
        if center is None:
            if mode[0] != "G01":
                out.append("G01*")
                mode[0] = "G01"
            out.append(f"{self.xy(b)}D01*")
            return
        want = "G02" if height > 0 else "G03"
        if mode[1] != "G75":
            out.append("G75*")
            mode[1] = "G75"
        if mode[0] != want:
            out.append(f"{want}*")
            mode[0] = want
        cx, cy, _r = center
        out.append(f"{self.xy(b)}I{self.num(cx - a[0])}J{self.num(cy - a[1])}D01*")

    def replacement(self, obj, record: _Record, pieces: Sequence[Dict]) -> str:
        out: List[str] = []
        mode = ["", ""]  # interpolation, quadrant mode as written here
        aperture_changed = False
        for piece in pieces:
            kind, pts = piece["kind"], [tuple(p) for p in piece["points"]]
            if kind == TRACE:
                width = float(piece["width_mm"])
                same = isinstance(getattr(obj, "aperture", None), CircleAperture) and \
                    abs(obj.aperture.equivalent_width(MM) - width) < 1e-7
                if not same:
                    out.append(f"D{self.aperture_for(width)}*")
                    aperture_changed = True
                elif aperture_changed and record.aperture is not None:
                    out.append(f"D{record.aperture}*")
                    aperture_changed = False
                out.append(f"{self.xy(pts[0])}D02*")
                heights = piece.get("heights") or [0.0]
                self.edge(pts[0], pts[1], float(heights[0]), out, mode)
            elif kind == REGION:
                heights = list(piece.get("heights") or []) + [0.0] * len(pts)
                out.append(f"{self.xy(pts[0])}D02*")
                for i in range(len(pts)):
                    self.edge(pts[i], pts[(i + 1) % len(pts)], float(heights[i]), out, mode)
            elif kind == PAD:
                out.append(f"{self.xy(pts[0])}D03*")
        # Restore the state the original statements left.
        if aperture_changed and record.aperture is not None:
            out.append(f"D{record.aperture}*")
        if mode[1] == "G75" and record.quadrant_single:
            out.append("G74*")
        out.append(f"{_MODE.get(record.mode, 'G01')}*")
        if record.point is not None:
            x, y = record.point
            out.append(f"X{self.settings.write_gerber_value(x)}Y{self.settings.write_gerber_value(y)}D02*")
        return "\n".join(out) + "\n"


_KINDS = {TRACE: (go.Line, go.Arc), REGION: (go.Region,), PAD: (go.Flash,)}


def _kind_matches(obj, kind: str) -> bool:
    """Whether a replacement can be written in place of the original's statements."""
    return isinstance(obj, _KINDS.get(kind, ()))


def _x2(text: str) -> str:
    """A net name as an X2 attribute value: its delimiters escaped."""
    out = text.replace("\\", "\\\\")
    for ch in ",*%":
        out = out.replace(ch, f"\\u{ord(ch):04X}")
    return out


def _flashes(layer: LayerFile, writer: _Writer, flashes: Sequence[Dict]) -> str:
    """Flashes appended after everything else (via pads and clearances)."""
    out = ["G04 Padstacks written by EMShrimulator*"]
    dark: Optional[bool] = None
    for flash in flashes:
        if "link" in flash:
            index = int(flash["link"])
            record = layer.records[index] if 0 <= index < len(layer.records) else None
            if record is None or record.aperture is None:
                continue
            code = record.aperture
        else:
            if float(flash.get("d") or 0) <= 0:
                continue
            code = writer.aperture_for(float(flash["d"]))
        want = bool(flash.get("dark", True))
        if want != dark:
            out.append("%LPD*%" if want else "%LPC*%")
            dark = want
        net = flash.get("net")
        if net:
            out.append(f"%TO.N,{_x2(str(net))}*%")
        out.append(f"D{code}*")
        out.append(f"{writer.xy((float(flash['x']), float(flash['y'])))}D03*")
        if net:
            out.append("%TD*%")
    if dark is False:
        out.append("%LPD*%")
    return "\n".join(out) + "\n"


def patch(layer: LayerFile, edits: Dict[int, Sequence[Dict]], extra: Sequence[Dict] = ()) -> str:
    """The layer's Gerber with edited objects replaced by their pieces ([] deletes) and `extra` flashes appended."""
    if not edits and not extra:
        return layer.text
    if not layer.editable:
        raise ValueError(f"This layer cannot be edited: {layer.reason}.")
    writer = _Writer(layer)
    chunks: List[Tuple[int, int, str]] = []
    for index, pieces in edits.items():
        if not 0 <= index < len(layer.objects):
            continue  # an edit of an object the file no longer has: left out
        record, obj = layer.records[index], layer.objects[index]
        if record is None or not layer.pieces[index].editable:
            continue
        pieces = [p for p in pieces if check_piece(p) is None and _kind_matches(obj, p["kind"])]
        start, end = layer.spans[record.first][0], layer.spans[record.last][1]
        chunks.append((start, end, writer.replacement(obj, record, pieces)))
    chunks.sort()
    text, out, pos = layer.text, [], 0
    for start, end, body in chunks:
        out.append(text[pos:start])
        out.append(body)
        pos = end
    out.append(text[pos:])
    patched = "".join(out)
    if extra:
        block = _flashes(layer, writer, extra)
        end = patched.rfind("M02*")
        patched = patched + block if end < 0 else patched[:end] + block + patched[end:]
    if writer.new_apertures:
        at = layer.first_draw if layer.first_draw is not None else 0
        defs = "".join(f"%ADD{n}C,{w:.6f}*%\n" for w, n in writer.new_apertures.items())
        # first_draw comes before every chunk, so it hasn't moved.
        patched = patched[:at] + "G04 Apertures added by EMShrimulator copper edits*\n" + defs + patched[at:]
    return patched


# ------------------------------------------------------------------ files

def original_path(fab_dir: Path, filename: str) -> Path:
    """The layer as imported: kept aside once it has been edited, else the file itself."""
    kept = fab_dir / ORIGINAL_DIR / filename
    return kept if kept.is_file() else fab_dir / filename


def edits_tag(edits: Dict[int, Sequence[Dict]], original: Path, extra: Sequence[Dict] = ()) -> str:
    stat = original.stat()
    payload = json.dumps({str(k): list(v) for k, v in sorted(edits.items())}, sort_keys=True)
    if extra:
        payload += "|" + json.dumps(list(extra), sort_keys=True)
    return hashlib.sha1(f"{stat.st_size}:{stat.st_mtime_ns}|{payload}".encode("utf-8")).hexdigest()


def _replace(tmp: Path, target: Path) -> None:
    """os.replace, retried while another reader holds the file on Windows."""
    for attempt in range(20):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sync_layer(fab_dir: Path, filename: str, bounds_mm: BBox, edits: Dict[int, Sequence[Dict]],
               extra: Sequence[Dict] = ()) -> bool:
    """Write fab/<filename> with these edits and flashes; True if the file changed."""
    target = fab_dir / filename
    kept_dir = fab_dir / ORIGINAL_DIR
    kept, stamp = kept_dir / filename, kept_dir / f"{filename}.edits"
    if not edits and not extra:
        if not kept.is_file():
            return False
        tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.part")
        shutil.copy2(kept, tmp)
        _replace(tmp, target)
        kept.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        return True
    if not kept.is_file():
        if not target.is_file():
            return False
        kept_dir.mkdir(exist_ok=True)
        shutil.copy2(target, kept)
    tag = edits_tag(edits, kept, extra)
    if stamp.is_file() and target.is_file() and stamp.read_text(encoding="utf-8").strip() == tag:
        return False
    text = patch(read_layer(kept, bounds_mm), edits, extra)
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.part")
    tmp.write_text(text, encoding="utf-8", newline="")
    _replace(tmp, target)
    stamp.write_text(tag, encoding="utf-8")
    return True


def drop_originals(fab_dir: Path, filename: Optional[str] = None) -> None:
    """Forget the kept originals (all, or one file's)."""
    kept_dir = fab_dir / ORIGINAL_DIR
    if filename is None:
        shutil.rmtree(kept_dir, ignore_errors=True)
        return
    (kept_dir / filename).unlink(missing_ok=True)
    (kept_dir / f"{filename}.edits").unlink(missing_ok=True)
