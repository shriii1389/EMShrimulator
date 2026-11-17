"""ODB++ reader: the copper, outline, holes and nets of one step.

Feature records: P (pad), L (line), A (arc), S ... SE (surface with OB/OS/OC/OE contours), T/B (text and
barcodes, skipped). eda/data refers to features by their 0-based index in a layer's file.
Geometry is kept in mm with arcs as arcs, for odb_convert to write out as Gerber.
"""
from __future__ import annotations

import gzip
import math
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

Point = Tuple[float, float]
# ("L", x, y) straight to (x, y); ("A", x, y, xc, yc, clockwise) arc to (x, y) about (xc, yc).
Segment = Tuple
INCH_TO_MM = 25.4
USER_SYMBOL_DEPTH = 4  # deeper user-symbol nesting is treated as a loop


class ODBError(RuntimeError):
    pass


# --- archive handling ---

def extract(archive: Path, dest: Path) -> Path:
    """Unpack an ODB++ archive and return the directory holding `matrix/`."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        root = archive
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        root = dest
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:  # pragma: no cover - older tarfile
                tf.extractall(dest)
        root = dest
    else:
        raise ODBError(f"{archive.name} is not a directory, zip or tar archive")

    # Case-insensitive like the rest of this module: some exporters write MATRIX/MATRIX, and
    # Linux filesystems (unlike Windows) would otherwise miss it.
    matrix = next((p for p in root.rglob("matrix", case_sensitive=False)
                   if p.is_file() and p.parent.name.lower() == "matrix"), None)
    if matrix is None:
        raise ODBError("no matrix/matrix file found — this does not look like an ODB++ product model")
    return matrix.parent.parent


def child(parent: Path, name: str) -> Path:
    """`parent / name`, matched case-insensitively."""
    exact = parent / name
    if exact.exists() or not parent.is_dir():
        return exact
    lowered = name.lower()
    return next((p for p in parent.iterdir() if p.name.lower() == lowered), exact)


def unlzw(data: bytes) -> bytes:
    """Decode Unix `compress` (.Z) data."""
    if len(data) < 3 or data[0] != 0x1F or data[1] != 0x9D:
        raise ODBError("not Unix-compressed data")
    max_bits = data[2] & 0x1F
    block_mode = bool(data[2] & 0x80)
    if not 9 <= max_bits <= 16:
        raise ODBError(f"compressed with {max_bits}-bit codes, which compress never writes")
    max_max_code = 1 << max_bits
    prefix = [0] * max_max_code
    suffix = list(range(256)) + [0] * (max_max_code - 256)
    out = bytearray()
    n_bits = 9
    max_code = (1 << n_bits) - 1
    free_ent = 257 if block_mode else 256
    old_code = -1
    fin_char = 0
    base = 3  # byte where the current group of codes starts
    pos = 0   # bits into it
    total = len(data)

    def align(bits: int) -> int:
        group = n_bits << 3
        return (bits - 1) + (group - (bits - 1 + group) % group)

    while True:
        if free_ent > max_code and n_bits < max_bits:
            pos = align(pos)
            base, pos = base + (pos >> 3), 0
            n_bits += 1
            max_code = max_max_code if n_bits == max_bits else (1 << n_bits) - 1
            continue
        if ((total - base) << 3) - (n_bits - 1) <= pos:
            break
        i = base + (pos >> 3)
        word = data[i] | (data[i + 1] << 8 if i + 1 < total else 0) | (data[i + 2] << 16 if i + 2 < total else 0)
        code = (word >> (pos & 7)) & ((1 << n_bits) - 1)
        pos += n_bits
        if old_code == -1:
            if code >= 256:
                raise ODBError("corrupt compressed data")
            out.append(code)
            fin_char = old_code = code
            continue
        if code == 256 and block_mode:
            prefix[:256] = [0] * 256
            free_ent = 256
            pos = align(pos)
            base, pos = base + (pos >> 3), 0
            n_bits = 9
            max_code = (1 << n_bits) - 1
            continue
        in_code = code
        stack = bytearray()
        if code >= free_ent:  # the KwKwK case: the code being defined right now
            if code > free_ent:
                raise ODBError("corrupt compressed data")
            stack.append(fin_char)
            code = old_code
        while code >= 256:
            stack.append(suffix[code])
            code = prefix[code]
        fin_char = suffix[code]
        stack.append(fin_char)
        stack.reverse()
        out += stack
        if free_ent < max_max_code:
            prefix[free_ent] = old_code
            suffix[free_ent] = fin_char
            free_ent += 1
        old_code = in_code
    return bytes(out)


def _read_text(path: Path) -> str:
    """Read a feature/matrix file, transparently decompressing the `.Z` and `.gz` variants."""
    path = child(path.parent, path.name) if path.parent.exists() else path
    for candidate in (path, Path(str(path) + ".Z"), Path(str(path) + ".z"), Path(str(path) + ".gz"),
                      Path(str(path) + ".gzip")):
        candidate = child(candidate.parent, candidate.name) if candidate.parent.exists() else candidate
        if candidate.is_file():
            data = candidate.read_bytes()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            elif data[:2] == b"\x1f\x9d":
                data = unlzw(data)
            return data.decode("utf-8", errors="replace")
    raise ODBError(f"missing ODB++ file: {path}")


# --- matrix ---

@dataclass
class ODBLayer:
    name: str
    row: int
    type: str          # SIGNAL / POWER_GROUND / DRILL / ROUT / SOLDER_MASK / ...
    context: str       # BOARD / MISC
    polarity: str      # POSITIVE / NEGATIVE
    side: str          # TOP / BOTTOM / INNER / NONE
    start_name: str = ""
    end_name: str = ""

    @property
    def is_copper(self) -> bool:
        return self.type in ("SIGNAL", "POWER_GROUND", "MIXED")

    @property
    def is_drill(self) -> bool:
        return self.type in ("DRILL", "ROUT")


def _parse_blocks(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """ODB++ `KIND { KEY=value … }` records as (KIND, fields)."""
    blocks: List[Tuple[str, Dict[str, str]]] = []
    current: Optional[Dict[str, str]] = None
    kind = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("{"):
            kind = line[:-1].strip().upper()
            current = {}
        elif line == "}":
            if current is not None:
                blocks.append((kind, current))
            current = None
        elif current is not None and "=" in line:
            k, _, v = line.partition("=")
            current[k.strip().upper()] = v.strip()
    return blocks


def read_matrix(root: Path) -> Tuple[List[ODBLayer], List[str]]:
    """(layers in stackup order, step names)."""
    text = _read_text(child(root, "matrix") / "matrix")
    layers: List[ODBLayer] = []
    steps: List[Tuple[int, str]] = []
    for kind, block in _parse_blocks(text):
        if "NAME" not in block:
            continue
        if kind == "STEP" or (not kind and "COL" in block):
            steps.append((int(block.get("COL", "0") or 0), block["NAME"]))
        elif kind == "LAYER" or (not kind and "TYPE" in block):
            layers.append(ODBLayer(
                name=block["NAME"],
                row=int(block.get("ROW", "0") or 0),
                type=block.get("TYPE", "").upper(),
                context=block.get("CONTEXT", "BOARD").upper(),
                polarity=block.get("POLARITY", "POSITIVE").upper(),
                side=block.get("SIDE", "NONE").upper(),
                start_name=block.get("START_NAME", ""),
                end_name=block.get("END_NAME", ""),
            ))
    layers.sort(key=lambda layer: layer.row)
    return layers, [name for _, name in sorted(steps)]


# --- placement ---

@dataclass(frozen=True)
class Affine:
    """x' = a·x + b·y + tx, y' = c·x + d·y + ty."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    @classmethod
    def place(cls, x: float, y: float, angle_cw_deg: float = 0.0, mirror: bool = False) -> "Affine":
        """ODB++ pad placement: mirror in X, rotate clockwise, then move to (x, y)."""
        t = math.radians(angle_cw_deg)
        cos, sin = math.cos(t), math.sin(t)
        if abs(cos) < 1e-12:
            cos = 0.0
        if abs(sin) < 1e-12:
            sin = 0.0
        m = -1.0 if mirror else 1.0
        return cls(cos * m, sin, -sin * m, cos, x, y)

    def __mul__(self, inner: "Affine") -> "Affine":
        """self after inner."""
        return Affine(self.a * inner.a + self.b * inner.c, self.a * inner.b + self.b * inner.d,
                      self.c * inner.a + self.d * inner.c, self.c * inner.b + self.d * inner.d,
                      self.a * inner.tx + self.b * inner.ty + self.tx, self.c * inner.tx + self.d * inner.ty + self.ty)

    def apply(self, x: float, y: float) -> Point:
        return self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty

    @property
    def flips(self) -> bool:
        return self.a * self.d - self.b * self.c < 0

    @property
    def quarter_turns(self) -> Optional[int]:
        """Whether a 90°-multiple rotation swaps width and height (0/1); None for other angles."""
        if abs(self.b) < 1e-9 and abs(self.c) < 1e-9:
            return 0
        if abs(self.a) < 1e-9 and abs(self.d) < 1e-9:
            return 1
        return None


IDENTITY = Affine()


# --- contours ---

@dataclass
class Contour:
    """A closed outline: from `start`, segments that end back at it."""

    start: Point
    segments: List[Segment]
    hole: bool = False

    def nodes(self) -> List[Point]:
        return [self.start] + [(s[1], s[2]) for s in self.segments]

    def transformed(self, t: Affine) -> "Contour":
        flip = t.flips
        segments = []
        for s in self.segments:
            x, y = t.apply(s[1], s[2])
            if s[0] == "L":
                segments.append(("L", x, y))
            else:
                xc, yc = t.apply(s[3], s[4])
                segments.append(("A", x, y, xc, yc, s[5] != flip))
        return Contour(t.apply(*self.start), segments, self.hole)

    def reversed(self) -> "Contour":
        nodes = self.nodes()
        segments = []
        for i in range(len(self.segments) - 1, -1, -1):
            s = self.segments[i]
            x, y = nodes[i]
            segments.append(("L", x, y) if s[0] == "L" else ("A", x, y, s[3], s[4], not s[5]))
        return Contour(nodes[-1], segments, self.hole)

    def samples(self) -> List[Point]:
        """Vertices plus a few points along each arc."""
        out = [self.start]
        x0, y0 = self.start
        for s in self.segments:
            if s[0] == "A":
                out.extend(arc_points(x0, y0, s[1], s[2], s[3], s[4], s[5], 16)[1:-1])
            out.append((s[1], s[2]))
            x0, y0 = s[1], s[2]
        return out

    def signed_area(self) -> float:
        pts = self.samples()
        return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


def arc_points(x1: float, y1: float, x2: float, y2: float, xc: float, yc: float, clockwise: bool,
               per_turn: int = 64) -> List[Point]:
    """Points along an arc, both ends included. Coincident ends are a full circle."""
    r = math.hypot(x1 - xc, y1 - yc)
    a1 = math.atan2(y1 - yc, x1 - xc)
    a2 = math.atan2(y2 - yc, x2 - xc)
    if clockwise:
        while a2 >= a1 - 1e-12:
            a2 -= 2 * math.pi
    else:
        while a2 <= a1 + 1e-12:
            a2 += 2 * math.pi
    n = max(2, int(per_turn * abs(a2 - a1) / (2 * math.pi)) + 1)
    return [(xc + r * math.cos(a1 + (a2 - a1) * i / n), yc + r * math.sin(a1 + (a2 - a1) * i / n))
            for i in range(n + 1)]


def _circle(r: float, cx: float = 0.0, cy: float = 0.0) -> Contour:
    return Contour((cx + r, cy), [("A", cx - r, cy, cx, cy, False), ("A", cx + r, cy, cx, cy, False)])


def _polygon(points: Sequence[Point]) -> Contour:
    return Contour(points[0], [("L", x, y) for x, y in list(points[1:]) + [points[0]]])


def _rounded_rect(w: float, h: float, r: float, corners: str = "1234", chamfer: bool = False) -> Contour:
    """ODB++ corners: 1 top right, 2 top left, 3 bottom left, 4 bottom right."""
    r = max(0.0, min(r, w / 2, h / 2))
    hw, hh = w / 2, h / 2
    # Each corner: (corner, previous point, next point, centre), counter-clockwise.
    spec = {"4": ((hw, -hh), (hw - r, -hh), (hw, -hh + r), (hw - r, -hh + r)),
            "1": ((hw, hh), (hw, hh - r), (hw - r, hh), (hw - r, hh - r)),
            "2": ((-hw, hh), (-hw + r, hh), (-hw, hh - r), (-hw + r, hh - r)),
            "3": ((-hw, -hh), (-hw, -hh + r), (-hw + r, -hh), (-hw + r, -hh + r))}
    segments: List[Segment] = []
    for key in "4123":
        corner, before, after, centre = spec[key]
        if key in corners and r > 0:
            segments.append(("L",) + before)
            segments.append(("L",) + after if chamfer else ("A", after[0], after[1], centre[0], centre[1], False))
        else:
            segments.append(("L",) + corner)
    start = (segments[-1][1], segments[-1][2])
    # Drop zero-length segments.
    kept, at = [], start
    for s in segments:
        if math.hypot(s[1] - at[0], s[2] - at[1]) > 1e-12:
            kept.append(s)
            at = (s[1], s[2])
    return Contour(start, kept)


def _numbers(text: str) -> List[float]:
    return [float(v) for v in re.findall(r"\d+(?:\.\d*)?|\.\d+", text)]


# --- symbols ---

@dataclass
class Symbol:
    """A pad or line symbol in mm about its origin. kind: round, rect, oval, donut, outline, user, none or unknown."""

    name: str
    kind: str
    width: float = 0.0
    height: float = 0.0
    inner: float = 0.0
    contours: List[Contour] = field(default_factory=list)
    features: List["Feature"] = field(default_factory=list)
    approximate: bool = False  # drawn as a simpler shape (a thermal as its ring)

    @property
    def stroke_width(self) -> float:
        return max(self.width, self.height)

    def outline(self) -> List[Contour]:
        """Its shape as contours about the origin (holes after their island)."""
        if self.kind == "round":
            return [_circle(self.width / 2)]
        if self.kind == "rect":
            return [_rounded_rect(self.width, self.height, 0.0)]
        if self.kind == "oval":
            return [_rounded_rect(self.width, self.height, min(self.width, self.height) / 2)]
        if self.kind == "donut":
            hole = _circle(self.inner / 2).reversed()
            hole.hole = True
            return [_circle(self.width / 2), hole]
        return list(self.contours)


def standard_symbol(name: str, size_scale: float) -> Symbol:
    """A standard ODB++ symbol by name; `size_scale` converts its mils or microns to mm."""
    n = name.strip().lower()

    def dims(pattern: str) -> Optional[List[float]]:
        m = re.fullmatch(pattern, n)
        return None if m is None else [float(g) * size_scale if g is not None else None for g in m.groups()]

    if n in ("null", "") or n.startswith("null"):
        return Symbol(name, "none")
    if (v := dims(r"r([\d.]+)")) is not None:
        return Symbol(name, "round", v[0], v[0])
    if (v := dims(r"s([\d.]+)")) is not None:
        return Symbol(name, "rect", v[0], v[0])
    m = re.fullmatch(r"rect([\d.]+)x([\d.]+)(?:x(r|c)([\d.]+)(?:x(\d+))?)?", n)
    if m:
        w, h = float(m.group(1)) * size_scale, float(m.group(2)) * size_scale
        if not m.group(3):
            return Symbol(name, "rect", w, h)
        r = float(m.group(4)) * size_scale
        return Symbol(name, "outline", w, h,
                      contours=[_rounded_rect(w, h, r, m.group(5) or "1234", chamfer=m.group(3) == "c")])
    if (v := dims(r"oval([\d.]+)x([\d.]+)")) is not None:
        return Symbol(name, "oval", v[0], v[1])
    if (v := dims(r"oval([\d.]+)")) is not None:
        return Symbol(name, "round", v[0], v[0])
    if (v := dims(r"donut_r([\d.]+)x([\d.]+)")) is not None:
        return Symbol(name, "donut", v[0], v[0], inner=v[1])
    if (v := dims(r"donut_s([\d.]+)x([\d.]+)")) is not None:
        hole = _rounded_rect(v[1], v[1], 0.0).reversed()
        hole.hole = True
        return Symbol(name, "outline", v[0], v[0], contours=[_rounded_rect(v[0], v[0], 0.0), hole])
    if (v := dims(r"di([\d.]+)x([\d.]+)")) is not None:
        w, h = v[0] / 2, v[1] / 2
        return Symbol(name, "outline", v[0], v[1], contours=[_polygon([(w, 0), (0, h), (-w, 0), (0, -h)])])
    if (v := dims(r"oct([\d.]+)x([\d.]+)x([\d.]+)")) is not None:
        return Symbol(name, "outline", v[0], v[1], contours=[_rounded_rect(v[0], v[1], v[2], chamfer=True)])
    if (v := dims(r"hex_l([\d.]+)x([\d.]+)x([\d.]+)")) is not None:
        w, h, r = v[0] / 2, v[1] / 2, v[2]
        return Symbol(name, "outline", v[0], v[1],
                      contours=[_polygon([(w, 0), (w - r, h), (-w + r, h), (-w, 0), (-w + r, -h), (w - r, -h)])])
    if (v := dims(r"hex_s([\d.]+)x([\d.]+)x([\d.]+)")) is not None:
        w, h, r = v[0] / 2, v[1] / 2, v[2]
        return Symbol(name, "outline", v[0], v[1],
                      contours=[_polygon([(0, h), (-w, h - r), (-w, -h + r), (0, -h), (w, -h + r), (w, h - r)])])
    if (v := dims(r"tri([\d.]+)x([\d.]+)")) is not None:
        w, h = v[0] / 2, v[1] / 2
        return Symbol(name, "outline", v[0], v[1], contours=[_polygon([(-w, -h), (w, -h), (0, h)])])
    if (v := dims(r"el([\d.]+)x([\d.]+)")) is not None:
        a, b = v[0] / 2, v[1] / 2
        pts = [(a * math.cos(2 * math.pi * i / 48), b * math.sin(2 * math.pi * i / 48)) for i in range(48)]
        return Symbol(name, "outline", v[0], v[1], contours=[_polygon(pts)])
    # Thermals are drawn as plain rings.
    if (v := dims(r"thr([\d.]+)x([\d.]+)x.*")) is not None:
        return Symbol(name, "donut", v[0], v[0], inner=v[1], approximate=True)
    m = re.fullmatch(r"(rc_ths|rc_tho|ths|s_ths|s_tho|s_thr|sr_ths)([\d.]+)x([\d.]+)(x.*)?", n)
    if m:
        values = [v * size_scale for v in _numbers(n[len(m.group(1)):])]
        if m.group(1).startswith("rc_"):  # width x height x angle x spokes x gap x air gap
            w, h, air = values[0], values[1], values[-1] if len(values) >= 6 else values[1] / 4
            outer, hole = _rounded_rect(w, h, 0.0), _rounded_rect(max(w - 2 * air, 0.0), max(h - 2 * air, 0.0), 0.0)
        else:  # square: outer x inner x ...
            w = h = values[0]
            outer, hole = _rounded_rect(w, w, 0.0), _rounded_rect(values[1], values[1], 0.0)
        hole = hole.reversed()
        hole.hole = True
        return Symbol(name, "outline", w, h, contours=[outer, hole], approximate=True)
    return Symbol(name, "unknown")


# --- features ---

@dataclass
class Pad:
    x: float
    y: float
    symbol: Symbol
    positive: bool
    placement: Affine
    net: Optional[str] = None


@dataclass
class Stroke:
    x1: float
    y1: float
    x2: float
    y2: float
    symbol: Symbol
    positive: bool
    arc: Optional[Tuple[float, float, bool]] = None  # (xc, yc, clockwise)
    net: Optional[str] = None


@dataclass
class Surface:
    contours: List[Contour]  # each island, then its holes
    positive: bool
    net: Optional[str] = None


Feature = Union[Pad, Stroke, Surface]
FEATURE_RECORDS = ("P", "L", "A", "S", "T", "B")


@dataclass
class LayerGeometry:
    name: str
    features: List[Feature] = field(default_factory=list)
    unknown_symbols: Set[str] = field(default_factory=set)
    approximate_symbols: Set[str] = field(default_factory=set)

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        xs: List[float] = []
        ys: List[float] = []
        for f in self.features:
            if isinstance(f, Surface):
                for contour in f.contours:
                    for x, y in contour.samples():
                        xs.append(x)
                        ys.append(y)
            elif isinstance(f, Stroke):
                r = f.symbol.stroke_width / 2
                xs.extend([f.x1 - r, f.x1 + r, f.x2 - r, f.x2 + r])
                ys.extend([f.y1 - r, f.y1 + r, f.y2 - r, f.y2 + r])
            else:
                r = f.symbol.stroke_width / 2
                xs.extend([f.x - r, f.x + r])
                ys.extend([f.y - r, f.y + r])
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)


def _tokens(line: str) -> List[str]:
    return line.split(";", 1)[0].split()


def _units(lines: Sequence[str]) -> float:
    """mm per coordinate unit. ODB++ defaults to inches."""
    for raw in lines:
        s = raw.strip().upper().replace(" ", "")
        if s.startswith("UNITS="):
            return 1.0 if s.endswith("MM") else INCH_TO_MM
        if s in ("UMM", "UINCH"):
            return 1.0 if s == "UMM" else INCH_TO_MM
    return INCH_TO_MM


def _orientation(tokens: Sequence[str], at: int) -> Tuple[float, bool]:
    """(clockwise degrees, mirrored) from an orient_def."""
    if len(tokens) <= at:
        return 0.0, False
    try:
        code = int(float(tokens[at]))
    except ValueError:
        return 0.0, False
    if code in (8, 9):
        angle = float(tokens[at + 1]) if len(tokens) > at + 1 else 0.0
        return angle, code == 9
    return 90.0 * (code % 4), code >= 4


class SymbolLibrary:
    """The product's symbols, user ones read once each."""

    def __init__(self, root: Optional[Path]):
        self.root = root
        self._user: Dict[str, Optional[List[Feature]]] = {}
        self._loading: Set[str] = set()

    def resolve(self, name: str, size_scale: float, depth: int = 0) -> Symbol:
        symbol = standard_symbol(name, size_scale)
        if symbol.kind != "unknown" or self.root is None or depth >= USER_SYMBOL_DEPTH:
            return symbol
        features = self._user_features(name, depth)
        if features is None:
            return symbol
        return Symbol(name, "user", features=features)

    def _user_features(self, name: str, depth: int) -> Optional[List[Feature]]:
        key = name.lower()
        if key in self._user:
            return self._user[key]
        if key in self._loading:  # a symbol that places itself
            return None
        folder = child(child(self.root, "symbols"), name)
        if not folder.is_dir():
            self._user[key] = None
            return None
        self._loading.add(key)
        try:
            geom = parse_features(_read_text(folder / "features"), self, depth=depth + 1)
            self._user[key] = geom.features
        except ODBError:
            self._user[key] = None
        finally:
            self._loading.discard(key)
        return self._user[key]


def parse_features(text: str, library: Optional[SymbolLibrary] = None, nets: Optional[Dict[int, str]] = None,
                   depth: int = 0) -> LayerGeometry:
    """A features file. `nets` maps feature index -> net name, from eda/data."""
    lines = text.splitlines()
    scale = _units(lines)
    size_scale = scale / 1000.0  # mils or microns -> mm
    library = library or SymbolLibrary(None)
    nets = nets or {}
    names: Dict[int, str] = {}
    symbols: Dict[int, Symbol] = {}
    geom = LayerGeometry(name="")

    def symbol(index_text: str) -> Symbol:
        try:
            index = int(float(index_text))
        except ValueError:
            return Symbol(index_text, "unknown")
        found = symbols.get(index)
        if found is None:
            found = library.resolve(names.get(index, ""), size_scale, depth)
            symbols[index] = found
            if found.kind == "unknown":
                geom.unknown_symbols.add(found.name or f"${index}")
            elif found.approximate:
                geom.approximate_symbols.add(found.name)
        return found

    feature = -1
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("$"):
            parts = raw[1:].split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                names[int(parts[0])] = parts[1].split(";", 1)[0].strip()
            continue
        t = _tokens(raw)
        if not t:
            continue
        rec = t[0].upper()
        if rec not in FEATURE_RECORDS:
            continue
        feature += 1
        net = nets.get(feature)
        try:
            if rec == "P" and len(t) >= 5:
                at = 3
                if t[3] == "-1" and len(t) >= 7:  # a resized symbol: -1 <sym> <factor>
                    at = 5
                    sym = symbol(t[4])
                else:
                    sym = symbol(t[3])
                angle, mirror = _orientation(t, at + 3)
                x, y = float(t[1]) * scale, float(t[2]) * scale
                geom.features.append(Pad(x, y, sym, t[at + 1].upper() != "N", Affine.place(x, y, angle, mirror), net))
            elif rec == "L" and len(t) >= 7:
                geom.features.append(Stroke(float(t[1]) * scale, float(t[2]) * scale, float(t[3]) * scale,
                                            float(t[4]) * scale, symbol(t[5]), t[6].upper() != "N", net=net))
            elif rec == "A" and len(t) >= 9:
                clockwise = t[10].upper().startswith("Y") if len(t) > 10 else t[-1].upper().startswith("Y")
                geom.features.append(Stroke(float(t[1]) * scale, float(t[2]) * scale, float(t[3]) * scale,
                                            float(t[4]) * scale, symbol(t[7]), t[8].upper() != "N",
                                            (float(t[5]) * scale, float(t[6]) * scale, clockwise), net=net))
            elif rec == "S":
                positive = len(t) < 2 or t[1].upper() != "N"
                contours, i = _read_surface(lines, i, scale)
                if contours:
                    geom.features.append(Surface(contours, positive, net))
        except (ValueError, IndexError):
            continue  # a malformed record is left out, not the whole layer
    return geom


def _read_surface(lines: Sequence[str], i: int, scale: float) -> Tuple[List[Contour], int]:
    contours: List[Contour] = []
    current: Optional[Contour] = None
    while i < len(lines):
        st = _tokens(lines[i].strip())
        i += 1
        if not st:
            continue
        op = st[0].upper()
        if op == "SE":
            break
        if op == "OB" and len(st) >= 3:
            current = Contour((float(st[1]) * scale, float(st[2]) * scale), [],
                              hole=len(st) >= 4 and st[3].upper() == "H")
        elif op == "OS" and len(st) >= 3 and current is not None:
            current.segments.append(("L", float(st[1]) * scale, float(st[2]) * scale))
        elif op == "OC" and len(st) >= 6 and current is not None:
            current.segments.append(("A", float(st[1]) * scale, float(st[2]) * scale, float(st[3]) * scale,
                                     float(st[4]) * scale, st[5].upper().startswith("Y")))
        elif op == "OE" and current is not None:
            if current.segments:
                last = current.segments[-1]
                if math.hypot(last[1] - current.start[0], last[2] - current.start[1]) > 1e-9:
                    current.segments.append(("L",) + current.start)  # close it
                contours.append(current)
            current = None
    return contours, i


# --- nets ---

def read_nets(root: Path, step: str) -> Dict[str, Dict[int, str]]:
    """{layer name (lower case): {feature index: net}} from eda/data; empty without one."""
    try:
        text = _read_text(child(child(child(root, "steps"), step), "eda") / "data")
    except ODBError:
        return {}
    layers: List[str] = []
    out: Dict[str, Dict[int, str]] = {}
    net: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("LYR "):
            layers.extend(name.lower() for name in line[4:].split(";", 1)[0].split())
        elif line.startswith("NET "):
            name = line[4:].split(";", 1)[0].strip()
            net = None if name in ("", "$NONE$") else name
        elif line.startswith("FID ") and net is not None:
            t = line.split()
            if len(t) >= 4 and t[1].upper() == "C":
                try:
                    layer = layers[int(t[2])]
                    out.setdefault(layer, {})[int(t[3])] = net
                except (ValueError, IndexError):
                    continue
    return out


# --- loading ---

def read_layer(root: Path, step: str, layer: str, library: Optional[SymbolLibrary] = None,
               nets: Optional[Dict[int, str]] = None) -> LayerGeometry:
    folder = child(child(child(child(root, "steps"), step), "layers"), layer)
    geom = parse_features(_read_text(folder / "features"), library or SymbolLibrary(root), nets)
    geom.name = layer
    return geom


def read_profile(root: Path, step: str) -> Optional[LayerGeometry]:
    """The board outline, from the step's `profile` file."""
    try:
        geom = parse_features(_read_text(child(child(root, "steps"), step) / "profile"))
    except ODBError:
        return None
    geom.name = "profile"
    return geom


def read_drills(root: Path, step: str, layers: Iterable[ODBLayer],
                library: Optional[SymbolLibrary] = None) -> List[Tuple[float, float, float]]:
    """(x_mm, y_mm, diameter_mm) for every round hole on every drill layer."""
    holes: List[Tuple[float, float, float]] = []
    library = library or SymbolLibrary(root)
    for layer in layers:
        if not layer.is_drill:
            continue
        try:
            geom = read_layer(root, step, layer.name, library)
        except ODBError:
            continue
        for f in geom.features:
            if isinstance(f, Pad) and f.symbol.kind == "round" and f.symbol.width > 0:
                holes.append((f.x, f.y, f.symbol.width))
    return holes


@dataclass
class ODBBoard:
    root: Path
    step: str
    steps: List[str]
    layers: List[ODBLayer]
    copper: List[LayerGeometry]
    outline: Optional[LayerGeometry]
    drills: List[Tuple[float, float, float]]
    copper_layers: List[ODBLayer] = field(default_factory=list)  # matching `copper`, one for one

    def bounds_mm(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        source = [self.outline] if self.outline and self.outline.features else self.copper
        xs: List[float] = []
        ys: List[float] = []
        for geom in source:
            b = geom.bounds()
            if b:
                xs.extend([b[0], b[2]])
                ys.extend([b[1], b[3]])
        if not xs:
            raise ODBError("board has no geometry to bound")
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    @property
    def net_names(self) -> List[str]:
        names = {f.net for geom in self.copper for f in geom.features if f.net}
        return sorted(names)


def _choose_step(steps: Sequence[str], wanted: Optional[str]) -> str:
    if wanted:
        match = next((s for s in steps if s.lower() == wanted.lower()), None)
        if match is None:
            raise ODBError(f"there is no step '{wanted}'; the product has {', '.join(steps)}")
        return match
    # A product often holds the board and a panel: pick the board.
    return next((s for s in steps if re.search(r"pcb|board", s, re.IGNORECASE)), steps[0])


def load(archive: Path, work_dir: Path, step: Optional[str] = None) -> ODBBoard:
    root = extract(archive, work_dir)
    layers, steps = read_matrix(root)
    steps_dir = child(root, "steps")
    present = {p.name.lower(): p.name for p in steps_dir.iterdir() if p.is_dir()} if steps_dir.is_dir() else {}
    steps = [present.get(s.lower(), s) for s in steps if s.lower() in present] or sorted(present.values())
    if not steps:
        raise ODBError("the product model contains no steps")
    chosen = _choose_step(steps, step)

    library = SymbolLibrary(root)
    nets = read_nets(root, chosen)
    copper, copper_layers = [], []
    for layer in layers:
        if layer.is_copper and layer.context == "BOARD":
            try:
                copper.append(read_layer(root, chosen, layer.name, library, nets.get(layer.name.lower())))
                copper_layers.append(layer)
            except ODBError:
                continue
    return ODBBoard(
        root=root, step=chosen, steps=list(steps), layers=layers, copper=copper,
        outline=read_profile(root, chosen),
        drills=read_drills(root, chosen, layers, library),
        copper_layers=copper_layers,
    )


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
