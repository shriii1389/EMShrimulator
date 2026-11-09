"""The results chart: one QPainter widget for every view, following the app theme.

Drag pans, the wheel zooms (Ctrl: x only, Shift: y only), Shift+drag zooms to a box, double-click resets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QWidget

from EMShrimulator.app import theme

Range = Tuple[float, float, float, float]   # x0, x1, y0, y1 in axis space (log10 of x on a log axis)

INTERACTION_HINT = ("Drag to pan · wheel to zoom (Ctrl: x only, Shift: y only) · Shift+drag to zoom to a box · "
                    "double-click to fit · right-click to copy or save")

# Magma, sampled at 33 even steps.
MAGMA = ("#000004 #030312 #0a0822 #130d34 #1d1147 #29115a #36106b #440f76 #51127c #5d177f #6a1c81 #762181 "
         "#832681 #902a81 #9c2e7f #aa337d #b73779 #c43c75 #d0416f #dc4869 #e75263 #ef5d5e #f56b5c #f9795d "
         "#fc8961 #fd9869 #fea772 #feb67c #fec488 #fed395 #fde2a3 #fcf0b2 #fcfdbf").split()

_PREFIXES = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"),
             (1e-9, "n"), (1e-12, "p")]
_SETTLE_MS = 140      # after the last pan/zoom step, redraw at full line width
_PICK_PX = 24.0       # how near the cursor a point must be for the readout


@dataclass
class _Line:
    x: np.ndarray
    y: np.ndarray
    color: str
    width: float
    dashed: bool
    label: Optional[str]
    guide: bool                        # grid-like decoration: no legend, readout or autorange
    marker: Optional[str]              # "o" or "s": a mark at every point instead of a line
    size: float
    readout: Optional[Callable[[int], str]]
    axis: str = "y1"                   # "y2": against the right-hand axis
    # Polylines in axis space, built once.
    polygons: Optional[List[QPolygonF]] = None
    polygons_log_x: bool = False


@dataclass
class _Text:
    x: float
    y: float
    text: str
    color: Optional[str]
    axes: bool                         # x, y as fractions of the plot area instead of data
    offset: Tuple[float, float]
    align: Qt.AlignmentFlag
    size: float


@dataclass
class _Image:
    image: QImage
    values: np.ndarray                 # (rows, cols), row 0 at extent y0
    extent: Tuple[float, float, float, float]
    vmin: float
    vmax: float
    label: str
    readout: Optional[Callable[[float, float, float], str]]


# ------------------------------------------------------------------ helpers

def _nice_step(span: float, count: float) -> float:
    raw = span / max(count, 1.0)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * magnitude:
            return m * magnitude
    return 10 * magnitude


def nice_ticks(lo: float, hi: float, count: float) -> Tuple[np.ndarray, float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return np.array([]), 1.0
    step = _nice_step(hi - lo, count)
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return np.arange(first, last + 1) * step, step


def _decimals(step: float) -> int:
    """Decimal places that show every multiple of `step` exactly."""
    if step <= 0:
        return 0
    places = max(0, -math.floor(math.log10(step) + 1e-9))
    while places < 9 and abs(step * 10 ** places - round(step * 10 ** places)) > 1e-6:
        places += 1
    return places


def _strip(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def eng(value: float, unit: str = "", step: Optional[float] = None) -> str:
    """2.45 GHz, 120 µm: a number with an SI prefix, to the precision `step` needs."""
    if value == 0 or not math.isfinite(value):
        return f"0 {unit}".strip() if value == 0 else "—"
    magnitude = abs(value)
    factor, prefix = next(((f, p) for f, p in _PREFIXES if magnitude >= f * 0.9995), _PREFIXES[-1])
    if step is not None:
        text = _strip(f"{value / factor:.{_decimals(step / factor)}f}")
    else:
        text = f"{value / factor:.4g}"
    return f"{text} {prefix}{unit}".strip()


def _plain(value: float, step: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-4:
        return f"{value:.3g}"
    return _strip(f"{value:.{_decimals(step)}f}")


def colormap_lut(stops: Sequence[str] = MAGMA) -> np.ndarray:
    """256 ARGB32 words interpolated through `stops`."""
    rgb = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    at = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    channels = [np.interp(t, at, rgb[:, c]).round().astype(np.uint32) for c in range(3)]
    return (0xFF000000 | (channels[0] << 16) | (channels[1] << 8) | channels[2]).astype(np.uint32)


def _polygons(px: np.ndarray, py: np.ndarray) -> List[QPolygonF]:
    """Polylines through the finite points, broken wherever a value is missing."""
    finite = np.isfinite(px) & np.isfinite(py)
    if finite.all():
        runs = [(0, len(px))]
    else:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(np.int8), [0]])))
        runs = list(zip(edges[::2], edges[1::2]))
    return [QPolygonF(list(map(QPointF, px[a:b].tolist(), py[a:b].tolist()))) for a, b in runs if b - a >= 1]


# ------------------------------------------------------------------ widget

class Chart(QWidget):
    """One set of axes. Fill it with clear() and the plot/text/set_* calls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(220, 160)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setStatusTip(INTERACTION_HINT)
        self._cache: Optional[QPixmap] = None
        self._fast = False
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._settled)
        self._drag: Optional[Tuple[str, QPointF, Range]] = None   # ("pan" | "box", start, view at start)
        self._box_end: Optional[QPointF] = None
        self._hover: Optional[Tuple[QPointF, str]] = None
        # Pixel geometry of the last render, for hit-testing without re-rendering.
        self._frame: Optional[Tuple[QRectF, Range]] = None
        self._picks: List[Tuple[np.ndarray, np.ndarray, _Line]] = []
        self.clear()
        # A bound method, so Qt drops the connection when this widget is deleted.
        theme.manager().changed.connect(self._theme_changed)

    def _theme_changed(self, _name: str) -> None:
        self.invalidate()

    # ------------------------------------------------------------ content

    def clear(self) -> None:
        """Remove everything and forget the zoom."""
        self._lines: List[_Line] = []
        self._texts: List[_Text] = []
        self._vlines: List[Tuple[float, str, float]] = []
        self._image: Optional[_Image] = None
        self._labels = ("", "", "")
        self._x_unit = ""
        self._y_unit = ""
        self._y2_label = ""
        self._y2_unit = ""
        self._y2_limits: Optional[Tuple[float, float]] = None
        self._log_x = False
        self._limits: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]] = (None, None)
        self._equal = False
        self._axes = True
        self._legend: Optional[str] = "best"
        self._view: Optional[Range] = None
        self._hover = None
        self.invalidate()

    def plot(self, x, y, color: str, *, width: float = 1.5, dashed: bool = False, label: Optional[str] = None,
             guide: bool = False, marker: Optional[str] = None, size: float = 4.0,
             readout: Optional[Callable[[int], str]] = None, axis: str = "y1") -> None:
        """A line through (x, y), or with `marker`, a mark at each point; `axis="y2"` against the right axis."""
        self._lines.append(_Line(np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel(), color,
                                 width, dashed, label, guide, marker, size, readout, axis))
        self.invalidate()

    def text(self, x: float, y: float, text: str, *, color: Optional[str] = None, axes: bool = False,
             offset: Tuple[float, float] = (0.0, 0.0), align=Qt.AlignLeft | Qt.AlignBottom, size: float = 8.0) -> None:
        """A label at a data point, or with `axes`, at a fraction of the plot area."""
        self._texts.append(_Text(float(x), float(y), text, color, axes, offset, align, size))
        self.invalidate()

    def vline(self, x: float, color: str, width: float = 1.0) -> None:
        """A dotted vertical marker across the plot."""
        self._vlines.append((float(x), color, width))
        self.invalidate()

    def image(self, values, extent: Tuple[float, float, float, float], *, vmin: float, vmax: float,
              label: str = "", readout: Optional[Callable[[float, float, float], str]] = None) -> None:
        """A colour-mapped grid under the lines, with a colour bar."""
        values = np.asarray(values, dtype=float)
        scale = max(vmax - vmin, 1e-12)
        index = (np.clip(np.nan_to_num((values - vmin) / scale), 0.0, 1.0) * 255).astype(np.uint8)
        pixels = colormap_lut()[index][::-1]  # QImage rows run top-down
        rows, cols = pixels.shape
        image = QImage(pixels.tobytes(), cols, rows, cols * 4, QImage.Format_RGB32).copy()
        self._image = _Image(image, values, tuple(float(v) for v in extent), vmin, vmax, label, readout)
        self.invalidate()

    def set_labels(self, x: str = "", y: str = "", title: str = "", y2: Optional[str] = None) -> None:
        self._labels = (x, y, title)
        if y2 is not None:
            self._y2_label = y2
        self.invalidate()

    def set_y2(self, label: str = "", unit: str = "", limits: Optional[Tuple[float, float]] = None) -> None:
        """The right-hand axis: label, readout unit and fixed range (None = fitted)."""
        self._y2_label, self._y2_unit, self._y2_limits = label, unit, limits
        self.invalidate()

    def set_x_unit(self, unit: str) -> None:
        """Engineering tick labels on x (`2.5 GHz`), and the unit in the readout."""
        self._x_unit = unit
        self.invalidate()

    def set_y_unit(self, unit: str) -> None:
        """The unit the readout appends to y values."""
        self._y_unit = unit

    def set_log_x(self, on: bool) -> None:
        self._log_x = bool(on)
        self._view = None
        self.invalidate()

    def set_limits(self, x: Optional[Tuple[float, float]] = None, y: Optional[Tuple[float, float]] = None) -> None:
        """Fixed axis ranges; None leaves that axis fitted to the data."""
        self._limits = (x, y)
        self._view = None
        self.invalidate()

    def set_equal_aspect(self, on: bool) -> None:
        self._equal = bool(on)
        self.invalidate()

    def set_axes_visible(self, on: bool) -> None:
        """Ticks, tick labels and the cartesian grid. Smith and polar charts draw their own."""
        self._axes = bool(on)
        self.invalidate()

    def set_legend(self, where: Optional[str]) -> None:
        """'best', 'upper right', 'upper left', 'lower left', 'lower right', or None for no legend."""
        self._legend = where
        self.invalidate()

    def draw(self) -> None:
        """Kept for callers of the matplotlib canvas: painting already follows every change."""
        self.update()

    # ------------------------------------------------------------- view

    def invalidate(self) -> None:
        self._cache = None
        self.update()

    def reset_view(self) -> None:
        self._view = None
        self.invalidate()

    def _has_y2(self) -> bool:
        return any(line.axis == "y2" and not line.guide for line in self._lines)

    def _home_y2(self) -> Optional[Tuple[float, float]]:
        """The right axis's fitted range, or None when nothing is drawn against it."""
        if not self._has_y2():
            return None
        if self._y2_limits is not None:
            return self._y2_limits
        ys = []
        for line in self._lines:
            if line.axis == "y2" and not line.guide:
                ok = np.isfinite(line.y) & np.isfinite(line.x)
                if self._log_x:
                    ok &= line.x > 0
                if ok.any():
                    ys.append(line.y[ok])
        if not ys:
            return 0.0, 1.0
        values = np.concatenate(ys)
        y0, y1 = float(values.min()), float(values.max())
        pad = (y1 - y0) * 0.05 or max(abs(y0) * 0.05, 1.0)
        return y0 - pad, y1 + pad

    def _y2_range(self, y0: float, y1: float) -> Optional[Tuple[float, float]]:
        """Keep the right axis matched to the left through pan and zoom."""
        home2 = self._home_y2()
        if home2 is None:
            return None
        _x0, _x1, h0, h1 = self._home()
        scale = (home2[1] - home2[0]) / max(h1 - h0, 1e-300)
        return home2[0] + (y0 - h0) * scale, home2[0] + (y1 - h0) * scale

    def _home(self) -> Range:
        xlim, ylim = self._limits
        xs, ys = [], []
        for line in self._lines:
            if line.guide:
                continue
            ok = np.isfinite(line.x) & np.isfinite(line.y)
            if self._log_x:
                ok &= line.x > 0
            if ok.any():
                xs.append(line.x[ok])
                if line.axis != "y2":
                    ys.append(line.y[ok])
        if self._image is not None:
            e = self._image.extent
            xs.append(np.array([e[0], e[1]]))
            ys.append(np.array([e[2], e[3]]))
        if xlim is not None:
            x0, x1 = xlim
        elif xs:
            values = np.concatenate(xs)
            x0, x1 = float(values.min()), float(values.max())
        else:
            x0, x1 = 0.0, 1.0
        if ylim is not None:
            y0, y1 = ylim
        elif ys:
            values = np.concatenate(ys)
            y0, y1 = float(values.min()), float(values.max())
            if self._image is None:  # an image fills its axes exactly, as a map should
                pad = (y1 - y0) * 0.05 or max(abs(y0) * 0.05, 1.0)
                y0, y1 = y0 - pad, y1 + pad
        else:
            y0, y1 = 0.0, 1.0
        if self._log_x:
            x0, x1 = math.log10(max(x0, 1e-300)), math.log10(max(x1, 1e-300))
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 <= y0:
            y0, y1 = y0 - 0.5, y1 + 0.5
        return x0, x1, y0, y1

    def _ranges(self, rect: QRectF) -> Range:
        x0, x1, y0, y1 = self._view or self._home()
        if self._equal and rect.width() > 0 and rect.height() > 0:
            # Equal aspect: widen whichever axis is short.
            per_px = max((x1 - x0) / rect.width(), (y1 - y0) / rect.height())
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = per_px * rect.width() / 2, per_px * rect.height() / 2
            x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
        return x0, x1, y0, y1

    def _ax(self, x):
        """Data x into axis space."""
        if self._log_x:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.asarray(x) > 0, np.log10(np.maximum(x, 1e-300)), np.nan)
        return x

    def _unax(self, value: float) -> float:
        return 10 ** value if self._log_x else value

    # ------------------------------------------------------------ painting

    def _fonts(self) -> Tuple[QFont, QFont, QFont]:
        base = QFont(self.font())
        ticks = QFont(base)
        ticks.setPointSizeF(8.0)
        title = QFont(base)
        title.setPointSizeF(9.0)
        title.setBold(True)
        return base, ticks, title

    def _layout(self, size: QSize) -> QRectF:
        """The plot area inside the margins."""
        _base, tick_font, title_font = self._fonts()
        fm = QFontMetricsF(tick_font)
        xlabel, ylabel, title = self._labels
        left = 10.0
        bottom = 8.0
        if self._axes:
            left += fm.horizontalAdvance("−000.00") + 6
            bottom += fm.height() + 4
        if ylabel:
            left += fm.height() + 4
        if xlabel:
            bottom += fm.height() + 2
        top = 10.0 + (QFontMetricsF(title_font).height() + 4 if title else 0.0)
        right = 14.0
        if self._image is not None:
            right += 18 + fm.horizontalAdvance("−000.0") + 8 + (fm.height() + 4 if self._image.label else 0)
        elif self._axes and self._has_y2():
            right += fm.horizontalAdvance("−000.00") + 4 + (fm.height() + 4 if self._y2_label else 0)
        return QRectF(left, top, max(size.width() - left - right, 10.0), max(size.height() - top - bottom, 10.0))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        if self._cache is None or self._cache.size() != self.size() * self.devicePixelRatioF():
            ratio = self.devicePixelRatioF()
            pixmap = QPixmap(self.size() * ratio)
            pixmap.setDevicePixelRatio(ratio)
            painter = QPainter(pixmap)
            self.render_into(painter, self.size(), fast=self._fast)
            painter.end()
            self._cache = pixmap
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        self._paint_overlay(painter)
        painter.end()

    def render_into(self, painter: QPainter, size: QSize, fast: bool = False) -> None:
        """Draw the chart at `size`; `fast` uses 1 px lines while panning."""
        c = theme.colors()
        base_font, tick_font, title_font = self._fonts()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(QRectF(0, 0, size.width(), size.height()), QColor(c["panel"]))
        rect = self._layout(size)
        ranges = self._ranges(rect)
        x0, x1, y0, y1 = ranges
        sx = rect.width() / (x1 - x0)
        sy = rect.height() / (y1 - y0)
        self._frame = (rect, ranges)

        def to_px(x, y):
            return rect.left() + (self._ax(x) - x0) * sx, rect.bottom() - (np.asarray(y) - y0) * sy

        # The right axis, when anything is drawn against it.
        y2r = self._y2_range(y0, y1)
        sy2 = rect.height() / (y2r[1] - y2r[0]) if y2r is not None and y2r[1] > y2r[0] else sy

        def to_px_of(line, x, y):
            if line.axis == "y2" and y2r is not None:
                return rect.left() + (self._ax(x) - x0) * sx, rect.bottom() - (np.asarray(y) - y2r[0]) * sy2
            return to_px(x, y)

        painter.fillRect(rect, QColor(c["sunken"]))
        painter.setFont(tick_font)
        fm = QFontMetricsF(tick_font)
        grid_pen = QPen(QColor(c["grid_minor"]), 1)

        # Ticks first, so the grid sits under the data.
        xticks: List[Tuple[float, str]] = []
        yticks: List[Tuple[float, str]] = []
        if self._axes:
            xticks = self._x_ticks(x0, x1, rect.width())
            values, step = nice_ticks(y0, y1, max(rect.height() / 42, 2))
            yticks = [(float(v), _plain(v, step)) for v in values]
            painter.setPen(grid_pen)
            for v, _label in xticks:
                px = rect.left() + (v - x0) * sx
                painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
            for v, _label in yticks:
                py = rect.bottom() - (v - y0) * sy
                painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))

        painter.save()
        painter.setClipRect(rect)
        if self._image is not None:
            e = self._image.extent
            ix0, iy0 = to_px(e[0], e[2])
            ix1, iy1 = to_px(e[1], e[3])
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(QRectF(QPointF(float(ix0), float(iy1)), QPointF(float(ix1), float(iy0))),
                              self._image.image)
        # Cached polygons through the painter's transform, with cosmetic pens.
        to_screen = QTransform(sx, 0.0, 0.0, -sy, rect.left() - x0 * sx, rect.bottom() + y0 * sy)
        to_screen2 = (QTransform(sx, 0.0, 0.0, -sy2, rect.left() - x0 * sx, rect.bottom() + y2r[0] * sy2)
                      if y2r is not None else to_screen)
        self._picks = []
        for line in self._lines:
            color = QColor(line.color)
            if line.marker:
                px, py = to_px_of(line, line.x, line.y)
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                r = line.size / 2 + 0.5
                for x, y in zip(np.asarray(px).tolist(), np.asarray(py).tolist()):
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    if line.marker == "s":
                        painter.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
                    else:
                        painter.drawEllipse(QPointF(x, y), r, r)
                painter.setBrush(Qt.NoBrush)
                continue
            if line.polygons is None or line.polygons_log_x != self._log_x:
                line.polygons = _polygons(np.asarray(self._ax(line.x), dtype=float), line.y)
                line.polygons_log_x = self._log_x
            pen = QPen(color, 1.0 if (fast or line.guide) else line.width)
            pen.setCosmetic(True)
            if line.dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.save()
            painter.setWorldTransform(to_screen2 if line.axis == "y2" else to_screen, True)
            for polygon in line.polygons:
                painter.drawPolyline(polygon)
            painter.restore()
            if not line.guide and (line.label or line.readout):
                px, py = to_px_of(line, line.x, line.y)
                self._picks.append((np.asarray(px, dtype=float), np.asarray(py, dtype=float), line))
        for x, color, width in self._vlines:
            pen = QPen(QColor(color), width, Qt.DotLine)
            painter.setPen(pen)
            px = float(to_px(x, 0.0)[0])
            painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
        painter.restore()

        for t in self._texts:
            painter.setFont(self._sized(base_font, t.size))
            painter.setPen(QColor(t.color or c["text_faint"]))
            if t.axes:
                px, py = rect.left() + t.x * rect.width(), rect.bottom() - t.y * rect.height()
            else:
                px, py = (float(v) for v in to_px(t.x, t.y))
            self._draw_text(painter, QPointF(px + t.offset[0], py + t.offset[1]), t.text, t.align)

        painter.setPen(QPen(QColor(c["line"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        painter.setFont(tick_font)
        painter.setPen(QColor(c["text_dim"]))
        for v, label in xticks:
            px = rect.left() + (v - x0) * sx
            self._draw_text(painter, QPointF(px, rect.bottom() + 3), label, Qt.AlignHCenter | Qt.AlignTop)
        for v, label in yticks:
            py = rect.bottom() - (v - y0) * sy
            self._draw_text(painter, QPointF(rect.left() - 4, py), label, Qt.AlignRight | Qt.AlignVCenter)
        if y2r is not None and self._axes and self._image is None:
            values2, step2 = nice_ticks(y2r[0], y2r[1], max(rect.height() / 42, 2))
            painter.setPen(QColor(c["text_dim"]))
            for v in values2:
                py = rect.bottom() - (float(v) - y2r[0]) * sy2
                painter.drawLine(QPointF(rect.right(), py), QPointF(rect.right() + 3, py))
                self._draw_text(painter, QPointF(rect.right() + 5, py), _plain(float(v), step2),
                                Qt.AlignLeft | Qt.AlignVCenter)
            if self._y2_label:
                painter.save()
                painter.translate(rect.right() + 8 + fm.horizontalAdvance("−000.00") + fm.height() / 2,
                                  rect.center().y())
                painter.rotate(90)
                self._draw_text(painter, QPointF(0, 0), self._y2_label, Qt.AlignCenter)
                painter.restore()

        xlabel, ylabel, title = self._labels
        if xlabel:
            y = rect.bottom() + (fm.height() + 4 if self._axes else 3)
            self._draw_text(painter, QPointF(rect.center().x(), y), xlabel, Qt.AlignHCenter | Qt.AlignTop)
        if ylabel:
            painter.save()
            painter.translate(6 + fm.height() / 2, rect.center().y())
            painter.rotate(-90)
            self._draw_text(painter, QPointF(0, 0), ylabel, Qt.AlignCenter)
            painter.restore()
        if title:
            painter.setFont(title_font)
            painter.setPen(QColor(c["text"]))
            self._draw_text(painter, QPointF(rect.center().x(), rect.top() - 4), title, Qt.AlignHCenter | Qt.AlignBottom)

        if self._image is not None:
            self._paint_colorbar(painter, rect, tick_font)
        if self._legend:
            self._paint_legend(painter, rect, tick_font)

    def _x_ticks(self, x0: float, x1: float, width: float) -> List[Tuple[float, str]]:
        count = max(width / 90, 2)
        if self._log_x:
            ticks = []
            for decade in range(math.floor(x0), math.ceil(x1) + 1):
                for m in ((1,) if x1 - x0 > count / 2 else (1, 2, 5)):
                    v = math.log10(m) + decade
                    if x0 - 1e-9 <= v <= x1 + 1e-9:
                        ticks.append((v, eng(m * 10.0 ** decade, self._x_unit)))
            if len(ticks) >= 2:
                return ticks
            values, step = nice_ticks(10 ** x0, 10 ** x1, count)
            return [(math.log10(v), eng(v, self._x_unit, step)) for v in values if v > 0]
        values, step = nice_ticks(x0, x1, count)
        if self._x_unit:
            return [(float(v), eng(v, self._x_unit, step)) for v in values]
        return [(float(v), _plain(v, step)) for v in values]

    @staticmethod
    def _sized(font: QFont, points: float) -> QFont:
        sized = QFont(font)
        sized.setPointSizeF(points)
        return sized

    @staticmethod
    def _draw_text(painter: QPainter, at: QPointF, text: str, align) -> None:
        box = QRectF(at.x() - 2000, at.y() - 2000, 4000, 4000)
        if align & Qt.AlignLeft:
            box.moveLeft(at.x())
        elif align & Qt.AlignRight:
            box.moveRight(at.x())
        if align & Qt.AlignTop:
            box.moveTop(at.y())
        elif align & Qt.AlignBottom:
            box.moveBottom(at.y())
        painter.drawText(box, int(align), text)

    def _paint_legend(self, painter: QPainter, rect: QRectF, font: QFont) -> None:
        entries, seen = [], set()
        for line in self._lines:
            if line.label and not line.guide and not line.marker and line.label not in seen:
                seen.add(line.label)
                entries.append(line)
        if not entries:
            return
        both = self._has_y2() and any(e.axis != "y2" for e in entries)

        def text(line) -> str:
            # Which axis a trace reads against, once there are two.
            return f"{line.label}   ▸ Y2" if both and line.axis == "y2" else line.label

        c = theme.colors()
        fm = QFontMetricsF(font)
        row = fm.height() + 2
        swatch = 22.0
        width = swatch + 12 + max(fm.horizontalAdvance(text(e)) for e in entries)
        height = row * len(entries) + 8
        corners = {
            "upper right": QPointF(rect.right() - width - 8, rect.top() + 8),
            "upper left": QPointF(rect.left() + 8, rect.top() + 8),
            "lower left": QPointF(rect.left() + 8, rect.bottom() - height - 8),
            "lower right": QPointF(rect.right() - width - 8, rect.bottom() - height - 8),
        }
        where = self._legend if self._legend in corners else self._best_corner(corners, width, height)
        box = QRectF(corners[where].x(), corners[where].y(), width, height)
        background = QColor(c["raised"])
        background.setAlpha(225)
        painter.setPen(QPen(QColor(c["line"]), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(box, 3, 3)
        painter.setBrush(Qt.NoBrush)
        painter.setFont(font)
        for i, line in enumerate(entries):
            y = box.top() + 4 + row * i + row / 2
            pen = QPen(QColor(line.color), 2)
            if line.dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(box.left() + 6, y), QPointF(box.left() + 6 + swatch, y))
            painter.setPen(QColor(c["text"]))
            self._draw_text(painter, QPointF(box.left() + swatch + 10, y), text(line), Qt.AlignLeft | Qt.AlignVCenter)

    def _best_corner(self, corners, width: float, height: float) -> str:
        """The corner whose legend would cover the fewest points."""
        if not self._picks:
            return "upper right"
        px = np.concatenate([p[0] for p in self._picks])
        py = np.concatenate([p[1] for p in self._picks])
        best, fewest = "upper right", None
        for name, corner in corners.items():
            inside = int(np.count_nonzero((px >= corner.x()) & (px <= corner.x() + width)
                                          & (py >= corner.y()) & (py <= corner.y() + height)))
            if fewest is None or inside < fewest:
                best, fewest = name, inside
        return best

    def _paint_colorbar(self, painter: QPainter, rect: QRectF, font: QFont) -> None:
        image = self._image
        c = theme.colors()
        bar = QRectF(rect.right() + 14, rect.top(), 14, rect.height())
        strip = colormap_lut()[::-1].tobytes()
        painter.drawImage(bar, QImage(strip, 1, 256, 4, QImage.Format_RGB32).copy())
        painter.setPen(QPen(QColor(c["line"]), 1))
        painter.drawRect(bar)
        painter.setFont(font)
        painter.setPen(QColor(c["text_dim"]))
        values, step = nice_ticks(image.vmin, image.vmax, max(rect.height() / 42, 2))
        span = max(image.vmax - image.vmin, 1e-12)
        for v in values:
            y = bar.bottom() - (v - image.vmin) / span * bar.height()
            painter.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + 3, y))
            self._draw_text(painter, QPointF(bar.right() + 5, y), _plain(float(v), step), Qt.AlignLeft | Qt.AlignVCenter)
        if image.label:
            fm = QFontMetricsF(font)
            painter.save()
            painter.translate(bar.right() + 8 + fm.horizontalAdvance("−000.0") + fm.height() / 2, bar.center().y())
            painter.rotate(-90)
            self._draw_text(painter, QPointF(0, 0), image.label, Qt.AlignCenter)
            painter.restore()

    def _paint_overlay(self, painter: QPainter) -> None:
        c = theme.colors()
        painter.setRenderHint(QPainter.Antialiasing)
        if self._drag is not None and self._drag[0] == "box" and self._box_end is not None:
            accent = QColor(c["accent"])
            painter.setPen(QPen(accent, 1, Qt.DashLine))
            fill = QColor(accent)
            fill.setAlpha(40)
            painter.setBrush(fill)
            painter.drawRect(QRectF(self._drag[1], self._box_end).normalized())
            return
        if self._hover is None:
            return
        point, text = self._hover
        painter.setPen(QPen(QColor(c["text"]), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(point, 4.5, 4.5)
        _base, font, _title = self._fonts()
        painter.setFont(font)
        fm = QFontMetricsF(font)
        box = QRectF(0, 0, fm.horizontalAdvance(text) + 12, fm.height() + 6)
        box.moveBottomLeft(point + QPointF(10, -8))
        if box.right() > self.width() - 2:
            box.moveRight(point.x() - 10)
        if box.top() < 2:
            box.moveTop(point.y() + 8)
        background = QColor(c["raised"])
        background.setAlpha(235)
        painter.setPen(QPen(QColor(c["line_strong"]), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor(c["text"]))
        painter.drawText(box, int(Qt.AlignCenter), text)

    # ------------------------------------------------------------ readout

    def _readout_at(self, pos: QPointF) -> Optional[Tuple[QPointF, str]]:
        if self._frame is None:
            return None
        rect, ranges = self._frame
        if not rect.contains(pos):
            return None
        best = None
        for px, py, line in self._picks:
            d = np.hypot(px - pos.x(), py - pos.y())
            if not d.size or not np.isfinite(d).any():
                continue
            i = int(np.nanargmin(d))
            if d[i] <= _PICK_PX and (best is None or d[i] < best[0]):
                best = (float(d[i]), i, line, QPointF(float(px[i]), float(py[i])))
        if best is not None:
            _d, i, line, point = best
            if line.readout is not None:
                text = line.readout(i)
            else:
                x = eng(float(line.x[i]), self._x_unit) if self._x_unit else f"{line.x[i]:.4g}"
                unit = self._y2_unit if line.axis == "y2" else self._y_unit
                text = f"{line.label + ': ' if line.label else ''}{line.y[i]:.4g} {unit} @ {x}".replace("  ", " ")
            return point, text
        image = self._image
        if image is not None:
            x0, x1, y0, y1 = ranges
            x = x0 + (pos.x() - rect.left()) / rect.width() * (x1 - x0)
            y = y0 + (rect.bottom() - pos.y()) / rect.height() * (y1 - y0)
            e = image.extent
            if e[0] <= x <= e[1] and e[2] <= y <= e[3]:
                rows, cols = image.values.shape
                col = min(int((x - e[0]) / max(e[1] - e[0], 1e-12) * cols), cols - 1)
                row = min(int((y - e[2]) / max(e[3] - e[2], 1e-12) * rows), rows - 1)
                value = float(image.values[row, col])
                return pos, image.readout(x, y, value) if image.readout else f"{value:.3g} at ({x:.3g}, {y:.3g})"
        return None

    # ------------------------------------------------------------- input

    def _interacting(self) -> None:
        """Draw fast lines while the view moves; settle back to full quality shortly after."""
        self._fast = True
        self._settle.start()
        self.invalidate()

    def _settled(self) -> None:
        if self._drag is not None and self._drag[0] == "pan":
            self._settle.start()
            return
        self._fast = False
        self.invalidate()

    def _current(self) -> Optional[Tuple[QRectF, Range]]:
        if self._frame is None:
            return None
        rect, _old = self._frame
        return rect, self._ranges(rect)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        current = self._current()
        if current is None:
            return
        rect, (x0, x1, y0, y1) = current
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return
        factor = 0.85 ** (delta / 120.0)
        pos = event.position()
        fx = min(max((pos.x() - rect.left()) / rect.width(), 0.0), 1.0)
        fy = min(max((rect.bottom() - pos.y()) / rect.height(), 0.0), 1.0)
        mods = event.modifiers()
        zoom_x = not (mods & Qt.ShiftModifier) or self._equal
        zoom_y = not (mods & Qt.ControlModifier) or self._equal
        if zoom_x:
            cx = x0 + fx * (x1 - x0)
            x0, x1 = cx - (cx - x0) * factor, cx + (x1 - cx) * factor
        if zoom_y:
            cy = y0 + fy * (y1 - y0)
            y0, y1 = cy - (cy - y0) * factor, cy + (y1 - cy) * factor
        hx0, hx1, hy0, hy1 = self._home()
        if (x1 - x0) < (hx1 - hx0) * 1e-6 or (y1 - y0) < (hy1 - hy0) * 1e-6:
            return  # deep enough; beyond this coordinates overflow
        self._view = (x0, x1, y0, y1)
        self._hover = None
        self._interacting()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        current = self._current()
        if event.button() != Qt.LeftButton or current is None:
            super().mousePressEvent(event)
            return
        kind = "box" if event.modifiers() & Qt.ShiftModifier else "pan"
        self._drag = (kind, event.position(), current[1])
        self._box_end = None
        self._hover = None
        self.setCursor(Qt.CrossCursor if kind == "box" else Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        pos = event.position()
        if self._drag is None:
            hover = self._readout_at(pos)
            if hover != self._hover:
                self._hover = hover
                self.update()
            return
        kind, start, (x0, x1, y0, y1) = self._drag
        if kind == "box":
            self._box_end = pos
            self.update()
            return
        rect = self._frame[0]
        dx = (pos.x() - start.x()) / rect.width() * (x1 - x0)
        dy = (pos.y() - start.y()) / rect.height() * (y1 - y0)
        self._view = (x0 - dx, x1 - dx, y0 + dy, y1 + dy)
        self._interacting()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        kind, start, (x0, x1, y0, y1) = self._drag
        self._drag = None
        self.unsetCursor()
        end = event.position()
        if kind == "box" and self._frame is not None and abs(end.x() - start.x()) > 6 and abs(end.y() - start.y()) > 6:
            rect = self._frame[0]

            def at(p: QPointF) -> Tuple[float, float]:
                return (x0 + (p.x() - rect.left()) / rect.width() * (x1 - x0),
                        y0 + (rect.bottom() - p.y()) / rect.height() * (y1 - y0))

            ax, ay = at(start)
            bx, by = at(end)
            self._view = (min(ax, bx), max(ax, bx), min(ay, by), max(ay, by))
            self.invalidate()
        self._box_end = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.reset_view()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._cache = None
        super().resizeEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        menu.addAction("Fit to data", self.reset_view)
        menu.addSeparator()
        menu.addAction("Copy image", lambda: QApplication.clipboard().setPixmap(self.grab()))
        menu.addAction("Save image…", self.save_image)
        menu.exec(event.globalPos())

    def save_image(self) -> None:
        path, chosen = QFileDialog.getSaveFileName(self, "Save chart", "chart.png",
                                                   "PNG image (*.png);;SVG drawing (*.svg)")
        if not path:
            return
        if path.lower().endswith(".svg") or chosen.startswith("SVG"):
            from PySide6.QtSvg import QSvgGenerator

            generator = QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(self.size())
            generator.setViewBox(QRectF(0, 0, self.width(), self.height()))
            painter = QPainter(generator)
            self.render_into(painter, self.size())
            painter.end()
        else:
            # Twice the screen resolution, so the file stays sharp in a report.
            ratio = 2.0
            pixmap = QPixmap(self.size() * ratio)
            pixmap.setDevicePixelRatio(ratio)
            painter = QPainter(pixmap)
            self.render_into(painter, self.size())
            painter.end()
            pixmap.save(path)
