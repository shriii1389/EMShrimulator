"""2D layout canvas: copper layers, snapping, port tools and differential pairs.

Scene units are raster pixels at base resolution: x_mm = px / px_per_mm, y_mm = height_mm - py / px_per_mm.
Zoomed out, each layer is drawn from its raster; zoomed in, from its vector shapes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QByteArray, QDataStream, QIODevice, QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsView,
)

from EMShrimulator.app import theme
from EMShrimulator.app.background import BackgroundRunner
from EMShrimulator.app.views.layout import palette
from EMShrimulator.app.views.layout.copper_tool import CopperTool
from EMShrimulator.core.layout.snap import EdgeHit, SnapHit, SnapSet

MAX_ZOOM = 60.0
MIN_ZOOM = 0.02
SNAP_RADIUS_PX = 14.0
CLICK_TOLERANCE_PX = 4  # a right press and release closer than this is a click
TOOLS = ("select", "point", "edge", "circuit", "pair", "via")


COPPER_OPACITY = 0.92  # stacked layers stay readable through each other
SMOOTH_AFTER_ZOOM_MS = 160  # fast scaling while zooming, smoothed once after
VECTOR_ABOVE = 1.0  # device px per raster px above which shapes are drawn
INTERACTIVE_FRAME_MS = 40.0  # slower vector frames are drawn off the UI thread
STROKE_TOLERANCE_MM = 1e-4  # how far a trace outline's curves may stray from the true offset
NET_PROBE_MS = 60  # delay before looking up the net under the cursor


def mask_image(width: int, height: int, coverage: bytes) -> QImage:
    """An 8-bit coverage mask as a QImage that owns its pixels."""
    return QImage(coverage, width, height, width, QImage.Format_Alpha8).copy()


def tinted(mask: QImage, color: str, opacity: float = 1.0) -> QImage:
    """A coverage mask filled with `color` at `opacity`. Thread-safe."""
    image = QImage(mask.size(), QImage.Format_ARGB32_Premultiplied)
    fill = QColor(color)
    fill.setAlphaF(opacity)
    image.fill(fill)
    painter = QPainter(image)
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawImage(0, 0, mask)
    painter.end()
    return image


@dataclass
class Shape:
    """One of a layer's vector paths, in board-local mm (Y up), ready to fill."""

    run: int
    dark: bool
    width: float  # trace width in mm; 0 for a pad or region
    bounds: QRectF
    path: QPainterPath
    net: int = -1  # index into the layer's net names; -1 for copper on no net


def decode_shapes(shapes: Dict) -> List[Shape]:
    """gerber_shapes' serialised paths as QPainterPaths, with traces pre-outlined. Thread-safe."""
    data = QByteArray(shapes["blob"].tobytes())  # the stream reads this buffer: it must outlive the stream
    stream = QDataStream(data, QIODevice.ReadOnly)
    stream.setByteOrder(QDataStream.LittleEndian)
    stroker = QPainterPathStroker()
    stroker.setCapStyle(Qt.RoundCap)
    stroker.setJoinStyle(Qt.RoundJoin)
    stroker.setCurveThreshold(STROKE_TOLERANCE_MM)
    out = []
    for row in shapes["index"].tolist():
        run, dark, width, x0, y0, x1, y1 = row[:7]
        net = int(row[7]) if len(row) > 7 else -1
        path = QPainterPath()
        stream >> path
        if width:
            stroker.setWidth(width)
            path = stroker.createStroke(path)
        out.append(Shape(int(run), dark > 0, width, QRectF(x0, y0, x1 - x0, y1 - y0), path, net))
    if stream.status() != QDataStream.Ok or not stream.atEnd():
        raise ValueError("layer shapes do not match their index")
    return out


def paint_shapes(painter: QPainter, shapes: List[Shape], color: QColor, visible_mm: QRectF,
                 net_colors: Optional[List[QColor]] = None) -> None:
    """Paint a layer's shapes that meet `visible_mm` in `color`, or per net from `net_colors`."""
    brush = QBrush(color)
    brushes = [QBrush(c) for c in net_colors] if net_colors else None
    run = -1
    for shape in shapes:
        if not shape.bounds.intersects(visible_mm):
            continue
        if shape.run != run:
            run = shape.run
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver if shape.dark
                                       else QPainter.CompositionMode_DestinationOut)
        painter.fillPath(shape.path, brushes[shape.net] if brushes is not None else brush)


# (shapes, all_dark, color, opacity, net_colors or None)
StackLayer = Tuple[List[Shape], bool, QColor, float, Optional[List[QColor]]]


def render_stack(layers: List[StackLayer], size: QSize, transform: QTransform, visible_mm: QRectF) -> QImage:
    """Composite layers bottom first; `transform` maps mm to image pixels.

    Thread-safe, but one QPainterPath must never be painted from two threads at once.
    """
    image = QImage(size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    buffer: Optional[QImage] = None
    out = QPainter(image)
    out.setRenderHint(QPainter.Antialiasing)
    for layer in layers:
        shapes, all_dark, color, opacity = layer[:4]
        net_colors = layer[4] if len(layer) > 4 else None
        if all_dark and opacity >= 1.0:
            out.save()
            out.setWorldTransform(transform)
            paint_shapes(out, shapes, color, visible_mm, net_colors)
            out.restore()
            continue
        if buffer is None:
            buffer = QImage(size, QImage.Format_ARGB32_Premultiplied)
        buffer.fill(Qt.transparent)
        painter = QPainter(buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setWorldTransform(transform)
        paint_shapes(painter, shapes, color, visible_mm, net_colors)
        painter.end()
        out.setOpacity(opacity)
        out.drawImage(0, 0, buffer)
        out.setOpacity(1.0)
    out.end()
    return image


def net_id_map(shapes: List[Shape], size: QSize, transform: QTransform) -> np.ndarray:
    """Net index per raster pixel: 0 none, 1 copper on no net, 2 + i for net i."""
    width, height = size.width(), size.height()
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setWorldTransform(transform)
    painter.setPen(Qt.NoPen)
    brushes: Dict[int, QBrush] = {}
    for shape in shapes:
        key = shape.net + 2 if shape.dark else 0  # a clear run takes the copper away
        brush = brushes.get(key)
        if brush is None:
            brush = brushes[key] = QBrush(QColor.fromRgb((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF))
        painter.fillPath(shape.path, brush)
    painter.end()
    ids = np.frombuffer(image.constBits(), np.uint32).reshape(height, image.bytesPerLine() // 4)[:, :width] & 0xFFFFFF
    ids = ids.astype(np.uint16 if int(ids.max(initial=0)) < 65536 else np.uint32)
    # Extend nets by one pixel to cover the antialiased edges.
    for _ in range(2):
        grown = ids.copy()
        for target, source in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                               ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                               ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            empty = grown[target] == 0
            grown[target][empty] = ids[source][empty]
        ids = grown
    return ids


def colorize(ids: np.ndarray, coverage: np.ndarray, colors: List[QColor], fallback: QColor,
             opacity: float) -> QImage:
    """A layer's raster in its nets' colours, shaped by its antialiased `coverage`."""
    lut = np.zeros((len(colors) + 1, 4), np.float32)  # by id: 0 unknown, 1 no net, 2 + i net i
    for i, color in enumerate([fallback, colors[-1]] + list(colors[:-1])):
        lut[i] = (color.blueF(), color.greenF(), color.redF(), 1.0)
    lut *= 255.0 * opacity  # premultiplied, in QImage's BGRA byte order
    ids = np.minimum(ids, len(lut) - 1)
    pixels = lut.astype(np.uint16)[ids]
    pixels = ((pixels * coverage[..., None].astype(np.uint16) + 127) // 255).astype(np.uint8)
    height, width = ids.shape
    return QImage(np.ascontiguousarray(pixels).data, width, height, width * 4,
                  QImage.Format_ARGB32_Premultiplied).copy()


@dataclass
class _Frame:
    """The copper of the whole view drawn from shapes off the UI thread, for views where that is slow."""

    rect: QRectF  # scene rect it covers, on device-pixel boundaries at `scale`
    scale: float  # device pixels per scene unit it was drawn at
    key: tuple  # the layers and colours it shows
    image: QImage


def _outside(outer: QRectF, inner: QRectF) -> List[QRectF]:
    """`outer` less `inner`, as up to four rects."""
    inner = inner & outer
    if inner.isEmpty():
        return [outer]
    out = []
    if inner.top() > outer.top():
        out.append(QRectF(outer.left(), outer.top(), outer.width(), inner.top() - outer.top()))
    if inner.bottom() < outer.bottom():
        out.append(QRectF(outer.left(), inner.bottom(), outer.width(), outer.bottom() - inner.bottom()))
    if inner.left() > outer.left():
        out.append(QRectF(outer.left(), inner.top(), inner.left() - outer.left(), inner.height()))
    if inner.right() < outer.right():
        out.append(QRectF(inner.right(), inner.top(), outer.right() - inner.right(), inner.height()))
    return out


def _no_snap_modifier() -> bool:
    return bool(QApplication.keyboardModifiers() & Qt.ControlModifier)


# ------------------------------------------------------------------ screen-space drawing helpers

def _unit(p: QPointF) -> QPointF:
    length = math.hypot(p.x(), p.y())
    return QPointF(p.x() / length, p.y() / length) if length > 1e-12 else QPointF(0.0, -1.0)


def _fonts() -> Tuple[QFont, QFont]:
    """The label's two faces: the name, bold, and a smaller line of detail."""
    base = QFont(QApplication.font())
    bold = QFont(base)
    bold.setPointSizeF(8.6)
    bold.setWeight(QFont.DemiBold)
    small = QFont(base)
    small.setPointSizeF(7.6)
    return bold, small


def _pill_size(text: str, detail: str, wave: bool) -> Tuple[float, float]:
    bold, small = _fonts()
    fm, fs = QFontMetricsF(bold), QFontMetricsF(small)
    width = fm.horizontalAdvance(text) + (13.0 if wave else 0.0)
    height = fm.height()
    if detail:
        width = max(width, fs.horizontalAdvance(detail))
        height += fs.height()
    return width + 12.0, height + 4.0


def _offset_beyond(direction: QPointF, size: Tuple[float, float], gap: float) -> QPointF:
    """How far along `direction` an upright box of `size` must sit for its near side to be `gap` away."""
    reach = abs(direction.x()) * size[0] / 2 + abs(direction.y()) * size[1] / 2
    return direction * (gap + reach)


def _wave(painter: QPainter, left: QPointF, width: float, amplitude: float) -> None:
    """One period of a sine, a source's mark."""
    path = QPainterPath(left)
    steps = 16
    for i in range(1, steps + 1):
        t = i / steps
        path.lineTo(left + QPointF(width * t, -amplitude * math.sin(2 * math.pi * t)))
    painter.drawPath(path)


def _draw_pill(painter: QPainter, centre: QPointF, text: str, detail: str, ink: palette.PortInk, *,
               filled: bool, ring: Optional[QColor] = None, wave: bool = False,
               fill: Optional[QColor] = None, text_color: Optional[QColor] = None, warn: bool = False) -> QRectF:
    """A rounded label centred on a device point; filled for a driven port."""
    bold, small = _fonts()
    w, h = _pill_size(text, detail, wave)
    rect = QRectF(centre.x() - w / 2, centre.y() - h / 2, w, h)
    radius = min(h / 2, 9.0)
    background = fill or (ink.ink if filled else ink.paper)
    foreground = text_color or (ink.halo if filled or fill is not None else ink.ink)
    if ring is not None:
        painter.setPen(QPen(ring, 2.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(-2.5, -2.5, 2.5, 2.5), radius + 2.5, radius + 2.5)
    painter.setPen(QPen(ink.ink if not filled and fill is None else background, 1.0))
    painter.setBrush(background)
    painter.drawRoundedRect(rect, radius, radius)
    fm = QFontMetricsF(bold)
    painter.setFont(bold)
    painter.setPen(foreground)
    top = rect.top() + 2.0
    line = QRectF(rect.left() + 6.0, top, rect.width() - 12.0, fm.height())
    painter.drawText(line, Qt.AlignLeft | Qt.AlignVCenter, text)
    if wave:
        pen = QPen(foreground, 1.4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        start = QPointF(line.left() + fm.horizontalAdvance(text) + 3.0, line.center().y())
        _wave(painter, start, 9.0, 2.6)
    if detail:
        fs = QFontMetricsF(small)
        painter.setFont(small)
        painter.setPen(foreground if fill is not None else (ink.halo if filled else ink.dim))
        painter.drawText(QRectF(rect.left() + 6.0, top + fm.height(), rect.width() - 12.0, fs.height()),
                         Qt.AlignLeft | Qt.AlignVCenter, detail)
    if warn:
        painter.setPen(QPen(ink.halo, 1.2))
        painter.setBrush(ink.warn)
        painter.drawEllipse(QPointF(rect.right() - 1.0, rect.top() + 1.0), 3.6, 3.6)
    return rect


def _haloed_line(painter: QPainter, a: QPointF, b: QPointF, ink: QColor, halo: QColor, width: float,
                 glow: Optional[QColor] = None) -> None:
    for color, w in ((glow, width + 7.0), (halo, width + 3.0), (ink, width)):
        if color is None:
            continue
        pen = QPen(color, w)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(a, b)


class SnapMarker(QGraphicsItem):
    """The glyph under the cursor showing what it would snap to. Constant size on screen."""

    SIZE = 6.0

    def __init__(self):
        super().__init__()
        self.kind = "vertex"
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(60)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        s = self.SIZE + 3
        return QRectF(-s, -s, 2 * s, 2 * s)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        c = QColor(theme.colors()["accent"])
        s = self.SIZE
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(c, 2))
        painter.setBrush(Qt.NoBrush)
        kind = self.kind
        if kind in ("via", "pad"):
            painter.drawEllipse(QPointF(0, 0), s, s)
            painter.drawPoint(QPointF(0, 0))
        elif kind == "face":
            painter.drawEllipse(QPointF(0, 0), s, s)
            painter.drawLine(QPointF(-s, 0), QPointF(s, 0))
            painter.drawLine(QPointF(0, -s), QPointF(0, s))
        elif kind == "edge_mid":
            painter.drawPolygon(QPolygonF([QPointF(0, -s), QPointF(s, s * 0.8), QPointF(-s, s * 0.8)]))
        elif kind == "trace_mid":
            painter.drawPolygon(QPolygonF([QPointF(0, -s), QPointF(s, 0), QPointF(0, s), QPointF(-s, 0)]))
        elif kind == "trace_end":
            painter.drawRect(QRectF(-s, -s, 2 * s, 2 * s))
            painter.drawLine(QPointF(-s, -s), QPointF(s, s))
        else:  # vertex, outline, free
            painter.drawRect(QRectF(-s * 0.8, -s * 0.8, 1.6 * s, 1.6 * s))


TERMINAL_R = 6.5  # a circuit port's terminal, in screen pixels


def _terminal(painter: QPainter, centre: QPointF, positive: bool, ink: palette.PortInk,
              glow: Optional[QColor]) -> None:
    """A circuit port terminal: + filled in ink, − hollow, both with the halo round them."""
    r = TERMINAL_R
    if glow is not None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(centre, r + 4.5, r + 4.5)
    painter.setPen(QPen(ink.halo, 2.4))
    painter.setBrush(ink.ink if positive else ink.paper)
    painter.drawEllipse(centre, r + 0.6, r + 0.6)
    painter.setPen(QPen(ink.ink, 1.5))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(centre, r, r)
    pen = QPen(ink.halo if positive else ink.ink, 1.6)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    k = r * 0.5
    painter.drawLine(centre - QPointF(k, 0), centre + QPointF(k, 0))
    if positive:
        painter.drawLine(centre - QPointF(0, k), centre + QPointF(0, k))


class TerminalHandle(QGraphicsObject):
    """The - terminal of a circuit port; dragging it moves only that terminal."""

    def __init__(self, port_item: "PortItem"):
        super().__init__(port_item)
        self.port_item = port_item
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(1)
        self._dragging = False
        self._snap: Optional[SnapHit] = None

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        r = (TERMINAL_R + 6.0) / max(self.port_item.canvas.zoom(), 1e-3)
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        item = self.port_item
        centre = painter.worldTransform().map(QPointF(0, 0))
        painter.save()
        painter.setWorldTransform(QTransform())
        painter.setRenderHint(QPainter.Antialiasing)
        _terminal(painter, centre, False, palette.PortInk(), item.glow())
        painter.restore()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.port_item.canvas.portSelected.emit(self.port_item.number)
        self._dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseReleaseEvent(event)
        canvas = self.port_item.canvas
        canvas.hide_snap()
        if self._dragging:
            self._dragging = False
            x_mm, y_mm = canvas.to_mm(self.scenePos())
            hit = self._snap
            layer = hit.layer if hit is not None and hit.layer >= 0 else None
            label = hit.label(canvas.layer_names()) if hit is not None else None
            canvas.terminalMoved.emit(self.port_item.number, "neg", x_mm, y_mm, -1 if layer is None else layer,
                                      label or "")

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        if change == QGraphicsItem.ItemPositionChange and self._dragging:
            canvas = self.port_item.canvas
            scene = self.port_item.mapToScene(value)
            snapped, hit = canvas.snap_scene(scene)
            self._snap = hit
            canvas.cursorMoved.emit(*canvas.to_mm(snapped))
            return self.port_item.mapFromScene(snapped)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.port_item.prepareGeometryChange()
            self.port_item.update()
        return super().itemChange(change, value)


class PortItem(QGraphicsObject):
    """One layout port: an edge port, a microstrip launch, or a circuit port's terminals."""

    def __init__(self, canvas: "BoardCanvas", port):
        super().__init__()
        self.canvas = canvas
        self.number = port.number
        self.port = port
        self.selected_port = False
        self.hovered = False
        self.pair: Optional[Tuple[int, str]] = None  # (pair index, "+" or "−") when in a differential pair
        self.pair_selected = False
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(20)
        self._dragging = False
        self._snap: Optional[SnapHit] = None
        self.neg_handle = TerminalHandle(self)
        self.neg_handle.setVisible(False)

    # --------------------------------------------------------- geometry

    def is_circuit(self) -> bool:
        return self.port.kind.value == "circuit" and self.port.neg_x_mm is not None

    def neg_local(self) -> QPointF:
        if not self.is_circuit():
            return QPointF(0, 0)
        if self.neg_handle._dragging:
            return self.neg_handle.pos()
        return self.canvas.to_scene(self.port.neg_x_mm, self.port.neg_y_mm) - \
            self.canvas.to_scene(self.port.x_mm, self.port.y_mm)

    def outward_scene(self) -> QPointF:
        """Away from the copper, in scene coordinates: where a pair's bracket and the label go."""
        if self.is_circuit():
            return QPointF(0.0, -1.0)
        return _unit(self.mapToScene(QPointF(0, 1)) - self.mapToScene(QPointF(0, 0)))

    def glow(self) -> Optional[QColor]:
        """The selection's colour under the glyph when selected (or its pair is), fainter on hover."""
        ink = palette.PortInk()
        if self.selected_port or self.pair_selected:
            color = QColor(ink.select)
            color.setAlpha(170 if self.selected_port else 110)
            return color
        if self.hovered or self.canvas.tool_target == self.number:
            color = QColor(ink.select)
            color.setAlpha(80)
            return color
        return None

    def _geometry(self, t: QTransform) -> Dict:
        """The glyph's geometry in device pixels for the item-to-device transform `t`."""
        a = t.map(QPointF(0, 0))
        along = t.map(QPointF(1, 0)) - a
        scale = math.hypot(along.x(), along.y())
        u = _unit(along)
        n = _unit(t.map(QPointF(0, -1)) - a)
        ppm = self.canvas.base_ppm
        half = self.port.width_um / 1000 * ppm * scale / 2
        # A launch keeps a minimum size so it can be grabbed when zoomed out.
        half = max(half, 1.5) if self.port.kind.value == "lumped" else max(half, 6.0)
        length = max(self.port.length_um / 1000 * ppm * scale, 14.0)
        text, detail = self._label_text()
        size = _pill_size(text, detail, self.port.excite)
        g = {"a": a, "u": u, "n": n, "half": half, "length": length, "text": text, "detail": detail,
             "label": True, "size": size}
        if self.is_circuit():
            q = t.map(self.neg_local())
            g["q"] = q
            d = q - a
            if math.hypot(d.x(), d.y()) < 1.0:
                perp = QPointF(0.0, -1.0)
                mid = a
                gap = TERMINAL_R + 8.0
            else:
                e = _unit(QPointF(-d.y(), d.x()))
                perp = e if e.y() < 0 or (abs(e.y()) < 1e-6 and e.x() > 0) else e * -1
                mid = (a + q) / 2
                gap = 12.0
            g["centre"] = mid + _offset_beyond(perp, size, gap)
        elif self.pair is not None and (partner := self.canvas.partner_item(self.number)) is not None:
            # In a pair the label goes beside the port, away from its partner.
            toward = t.map(self.mapFromScene(partner.pos())) - a
            side = u * -1 if toward.x() * u.x() + toward.y() * u.y() > 0 else u
            g["centre"] = a + side * half + _offset_beyond(side, size, 8.0)
        else:
            g["centre"] = a + _offset_beyond(n * -1, size, 10.0)
        return g

    def _device_bounds(self, g: Dict) -> QRectF:
        points = [g["a"]]
        a, u, n = g["a"], g["u"], g["n"]
        if self.is_circuit():
            points.append(g["q"])
        else:
            points += [a + u * g["half"], a - u * g["half"], a + n * (g["length"] + 4), a - n * 18]
        # QRectF.united skips empty rects, so bound the points directly.
        xs, ys = [p.x() for p in points], [p.y() for p in points]
        rect = QRectF(min(xs) - 16, min(ys) - 16, max(xs) - min(xs) + 32, max(ys) - min(ys) + 32)
        if g["label"]:
            w, h = g["size"]
            rect = rect.united(QRectF(g["centre"].x() - w / 2 - 6, g["centre"].y() - h / 2 - 6, w + 12, h + 12))
        return rect

    def _device_transform(self) -> QTransform:
        return self.deviceTransform(self.canvas.viewportTransform())

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        t = self._device_transform()
        inverse, ok = t.inverted()
        if not ok:
            return QRectF()
        return inverse.mapRect(self._device_bounds(self._geometry(t)))

    def shape(self) -> QPainterPath:
        t = self._device_transform()
        inverse, ok = t.inverted()
        path = QPainterPath()
        if not ok:
            return path
        path.setFillRule(Qt.WindingFill)
        g = self._geometry(t)
        a, u, n = g["a"], g["u"], g["n"]
        path.addEllipse(a, 10.0, 10.0)
        stroker = QPainterPathStroker()
        stroker.setWidth(12.0)
        stroker.setCapStyle(Qt.RoundCap)
        if self.is_circuit():
            line = QPainterPath(a)
            line.lineTo(g["q"])
            path.addPath(stroker.createStroke(line))
        else:
            bar = QPainterPath(a - u * g["half"])
            bar.lineTo(a + u * g["half"])
            path.addPath(stroker.createStroke(bar))
            if self.port.kind.value != "lumped":
                launch = QPolygonF([a - u * g["half"], a + u * g["half"], a + u * g["half"] + n * g["length"],
                                    a - u * g["half"] + n * g["length"]])
                path.addPolygon(launch)
                path.closeSubpath()
        if g["label"]:
            w, h = g["size"]
            path.addRect(QRectF(g["centre"].x() - w / 2, g["centre"].y() - h / 2, w, h))
        return inverse.map(path)

    # ------------------------------------------------------------ paint

    def _layer_text(self) -> str:
        names = self.canvas.layer_names()

        def name(i: int) -> str:
            return names[i] if 0 <= i < len(names) else f"L{i}"

        p = self.port
        if self.is_circuit():
            return f"{name(p.layer)} + → {name(p.plane)} −"
        choice = self.canvas.references.get(p.number) if p.reference == "auto" else None
        plane = choice.plane if choice is not None and choice.plane is not None else p.plane
        return f"{name(p.layer)} → {name(plane)}" + (" (auto)" if p.reference == "auto" else "")

    def _label_text(self) -> Tuple[str, str]:
        p = self.port
        text = f"P{p.number + 1}"
        if not (self.selected_port or self.hovered):
            return text, ""
        if p.name and p.name != f"P{p.number + 1}":
            text += f"  {p.name}"
        kind = {"lumped": "edge", "microstrip": "launch", "circuit": "circuit"}.get(p.kind.value, p.kind.value)
        detail = f"{kind} · {self._layer_text()} · {p.impedance:g} Ω"
        if self.pair is not None and 0 <= self.pair[0] < len(self.canvas._pair_items):
            detail += f" · {self.pair[1]} of {self.canvas._pair_items[self.pair[0]].pair.name}"
        return text, detail

    def _warn(self) -> bool:
        p = self.port
        choice = self.canvas.references.get(p.number) if p.reference == "auto" else None
        return choice is not None and not choice.found

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        t = painter.worldTransform()
        g = self._geometry(t)
        ink = palette.PortInk()
        glow = self.glow()
        painter.save()
        painter.setWorldTransform(QTransform())
        painter.setRenderHint(QPainter.Antialiasing)
        kind = self.port.kind.value
        if self.is_circuit():
            self._paint_circuit(painter, g, ink, glow)
        elif kind == "lumped":
            self._paint_edge(painter, g, ink, glow)
        else:
            self._paint_launch(painter, g, ink, glow)
        if g["label"]:
            _draw_pill(painter, g["centre"], g["text"], g["detail"], ink, filled=self.port.excite,
                       ring=QColor(ink.select) if self.selected_port else None, wave=self.port.excite,
                       warn=self._warn())
        painter.restore()

    def _bar(self, painter: QPainter, g: Dict, ink: palette.PortInk, glow: Optional[QColor], width: float) -> None:
        """The port's sheet seen edge-on, across the copper edge, with a serif at each end for its span."""
        a, u, n, half = g["a"], g["u"], g["n"], g["half"]
        _haloed_line(painter, a - u * half, a + u * half, ink.ink, ink.halo, width, glow)
        for end in (a - u * half, a + u * half):
            _haloed_line(painter, end - n * 4.5, end + n * 4.5, ink.ink, ink.halo, 1.5)

    def _paint_edge(self, painter: QPainter, g: Dict, ink: palette.PortInk, glow: Optional[QColor]) -> None:
        """The edge the port spans, and nothing else: whether it is driven is on its label."""
        self._bar(painter, g, ink, glow, 4.0)

    def _paint_launch(self, painter: QPainter, g: Dict, ink: palette.PortInk, glow: Optional[QColor]) -> None:
        a, u, n, half, length = g["a"], g["u"], g["n"], g["half"], g["length"]
        body = QPolygonF([a - u * half, a + u * half, a + u * half + n * length, a - u * half + n * length])
        tint = QColor(ink.ink)
        tint.setAlpha(34)
        outline = QColor(ink.ink)
        outline.setAlpha(150)
        pen = QPen(outline, 1.0, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(tint)
        painter.drawPolygon(body)
        self._bar(painter, g, ink, glow, 3.4)

    def _paint_circuit(self, painter: QPainter, g: Dict, ink: palette.PortInk, glow: Optional[QColor]) -> None:
        a, q = g["a"], g["q"]
        d = q - a
        distance = math.hypot(d.x(), d.y())
        if distance > 2 * TERMINAL_R + 2:
            k = _unit(d)
            start, end = a + k * (TERMINAL_R + 1), q - k * (TERMINAL_R + 1)
            symbol = distance > 46.0
            mid = (a + q) / 2
            if symbol:
                _haloed_line(painter, start, mid - k * 9.5, ink.ink, ink.halo, 1.6, glow)
                _haloed_line(painter, mid + k * 9.5, end, ink.ink, ink.halo, 1.6, glow)
                self._source(painter, mid, k, ink, glow)
            else:
                _haloed_line(painter, start, end, ink.ink, ink.halo, 1.6, glow)
        _terminal(painter, a, True, ink, glow)

    def _source(self, painter: QPainter, centre: QPointF, k: QPointF, ink: palette.PortInk,
                glow: Optional[QColor]) -> None:
        """Between the terminals: a source (a sine in a circle) when driven, a load (a resistor) when not."""
        r = 9.0
        if glow is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(centre, r + 4, r + 4)
        painter.setPen(QPen(ink.halo, 3.0))
        painter.setBrush(ink.paper)
        painter.drawEllipse(centre, r, r)
        painter.setPen(QPen(ink.ink, 1.4))
        painter.drawEllipse(centre, r, r)
        e = QPointF(-k.y(), k.x())
        pen = QPen(ink.ink, 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        if self.port.excite:
            steps = 16
            for i in range(steps + 1):
                s = i / steps
                p = centre + k * (-5.5 + 11.0 * s) + e * (2.8 * math.sin(2 * math.pi * s))
                path.moveTo(p) if i == 0 else path.lineTo(p)
        else:
            zig = [-5.5, -3.7, -1.8, 0.0, 1.8, 3.7, 5.5]
            for i, s in enumerate(zig):
                side = 0.0 if i in (0, len(zig) - 1) else (2.6 if i % 2 else -2.6)
                p = centre + k * s + e * side
                path.moveTo(p) if i == 0 else path.lineTo(p)
        painter.drawPath(path)

    # ------------------------------------------------------------ input

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.prepareGeometryChange()
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.prepareGeometryChange()
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.canvas.portSelected.emit(self.number)
        self._dragging = True
        self._snap = None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseReleaseEvent(event)
        self.canvas.hide_snap()
        if self._dragging:
            self._dragging = False
            x_mm, y_mm = self.canvas.to_mm(self.pos())
            if (abs(x_mm - self.port.x_mm) > 1e-5) or (abs(y_mm - self.port.y_mm) > 1e-5):
                hit = self._snap
                label = hit.label(self.canvas.layer_names()) if hit is not None else ""
                self.canvas.portMoved.emit(self.number, x_mm, y_mm, label)

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        if change == QGraphicsItem.ItemPositionChange and self._dragging:
            snapped, hit = self.canvas.snap_scene(value)
            self._snap = hit
            self.canvas.cursorMoved.emit(*self.canvas.to_mm(snapped))
            return snapped
        if change == QGraphicsItem.ItemPositionHasChanged and self.pair is not None:
            self.canvas.update_pairs()
        return super().itemChange(change, value)


class PairItem(QGraphicsObject):
    """A differential pair: a bracket joining its two ports, labelled with the pair's name."""

    REACH = 17.0  # from a port's anchor to the bracket's start

    def __init__(self, canvas: "BoardCanvas", index: int, pair):
        super().__init__()
        self.canvas = canvas
        self.index = index
        self.pair = pair
        self.selected_pair = False
        self.hovered = False
        self.setZValue(18)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

    def _ports(self) -> Optional[Tuple[PortItem, PortItem]]:
        items = self.canvas._port_items
        p, n = items.get(self.pair.positive), items.get(self.pair.negative)
        return (p, n) if p is not None and n is not None else None

    def _geometry(self, t: QTransform) -> Optional[Dict]:
        ports = self._ports()
        if ports is None:
            return None
        ends, outs = [], []
        for item in ports:
            anchor = item.pos()
            a = t.map(anchor)
            o = _unit(t.map(anchor + item.outward_scene()) - a)
            ends.append(a + o * self.REACH)
            outs.append(o)
        start, end = ends
        o = outs[0] + outs[1]
        if math.hypot(o.x(), o.y()) < 0.3:  # facing away from each other: bow off the line between
            d = end - start
            o = QPointF(-d.y(), d.x())
            if o.y() > 0:
                o = o * -1
        o = _unit(o)
        span = math.hypot(end.x() - start.x(), end.y() - start.y())
        apex = (start + end) / 2 + o * (6.0 + min(0.08 * span, 18.0))
        control = apex * 2 - (start + end) / 2
        path = QPainterPath(start)
        path.quadTo(control, end)
        size = _pill_size(self.pair.name, "", False)
        # The name sits on the bracket's apex.
        return {"start": start, "end": end, "path": path, "centre": apex, "size": size}

    def _device_transform(self) -> QTransform:
        return self.deviceTransform(self.canvas.viewportTransform())

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        t = self._device_transform()
        g = self._geometry(t)
        inverse, ok = t.inverted()
        if g is None or not ok:
            return QRectF()
        w, h = g["size"]
        rect = g["path"].boundingRect().adjusted(-12, -12, 12, 12)
        rect = rect.united(QRectF(g["centre"].x() - w / 2 - 6, g["centre"].y() - h / 2 - 6, w + 12, h + 12))
        return inverse.mapRect(rect)

    def shape(self) -> QPainterPath:
        t = self._device_transform()
        g = self._geometry(t)
        inverse, ok = t.inverted()
        out = QPainterPath()
        if g is None or not ok:
            return out
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        out = stroker.createStroke(g["path"])
        out.setFillRule(Qt.WindingFill)
        w, h = g["size"]
        out.addRect(QRectF(g["centre"].x() - w / 2, g["centre"].y() - h / 2, w, h))
        for p in (g["start"], g["end"]):
            out.addEllipse(p, 7.0, 7.0)
        return inverse.map(out)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        g = self._geometry(painter.worldTransform())
        if g is None:
            return
        ink = palette.PortInk()
        color = palette.pair_color(self.index)
        painter.save()
        painter.setWorldTransform(QTransform())
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        layers = [(ink.halo, 5.0), (color, 2.0)]
        if self.selected_pair or self.hovered:
            glow = QColor(ink.select)
            glow.setAlpha(170 if self.selected_pair else 80)
            layers.insert(0, (glow, 10.0))
        for pen_color, width in layers:
            pen = QPen(pen_color, width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPath(g["path"])
        for point, sign in ((g["start"], "+"), (g["end"], "−")):
            painter.setPen(QPen(ink.halo, 2.0))
            painter.setBrush(color)
            painter.drawEllipse(point, 6.4, 6.4)
            pen = QPen(ink.halo, 1.6)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(point - QPointF(3.0, 0), point + QPointF(3.0, 0))
            if sign == "+":
                painter.drawLine(point - QPointF(0, 3.0), point + QPointF(0, 3.0))
        _draw_pill(painter, g["centre"], self.pair.name, "", ink, filled=True, fill=color,
                   ring=QColor(ink.select) if self.selected_pair else None)
        painter.restore()

    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton and self.canvas.tool == "select":
            self.canvas.pairSelected.emit(self.index)
            event.accept()
            return
        event.ignore()


class BoardCanvas(QGraphicsView):
    placeRequested = Signal(float, float, int, str)            # x, y, layer (-1 = unknown), snap label
    edgePortRequested = Signal(object, str)                    # EdgeHit, label
    circuitPortRequested = Signal(object, object, str, str)    # (x, y, layer) +, (x, y, layer) -, labels
    pairRequested = Signal(int, int)                           # + port number, - port number
    viaRequested = Signal(float, float, str)                   # x, y, snap label: the via tool's click
    portSelected = Signal(int)
    pairSelected = Signal(int)
    portMoved = Signal(int, float, float, str)
    terminalMoved = Signal(int, str, float, float, int, str)
    cursorMoved = Signal(float, float)
    snapChanged = Signal(str)
    netHovered = Signal(str)                                   # "GND · F_Cu", or "" off copper
    netActivated = Signal(str)                                 # a net double-clicked on the board
    toolFinished = Signal()
    emptyClicked = Signal()
    renderingChanged = Signal(bool)
    contextRequested = Signal(int, QPoint)  # port index under a right click (-1 for none), screen position
    pairMenuRequested = Signal(int, QPoint)  # a right click on a differential pair's bracket
    copperMenuRequested = Signal(object, object, object, QPoint)  # piece ref, handle (or None), (x, y) mm, where

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        # No scroll bars; a right or middle drag pans.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)

        self.board: Optional[Dict] = None
        self.base_ppm = 20.0
        self.height_mm = 0.0
        self.tool = "select"
        self.tool_target: Optional[int] = None  # the port the pair tool would take, under the cursor
        self.snapping = True
        self.snaps: SnapSet = SnapSet.empty()
        self.references: Dict = {}
        self.show_drills = True
        self.visible_layers: Dict[str, bool] = {}
        self.color_by = "layer"  # or "net"
        self.highlight: Optional[str] = None  # a net drawn as it is, the rest faded
        self._net_hex: Dict[str, str] = {}  # the board's nets and their colours
        # Layer name -> base-resolution raster; insertion order is drawing order.
        self._layers: Dict[str, Optional[QPixmap]] = {}
        # Layer name -> (UI-thread shapes, background-thread copy, all_dark).
        self._shapes: Dict[str, Tuple[List[Shape], List[Shape], bool]] = {}
        self._raw_shapes: Dict[str, Dict] = {}  # as loaded, to decode again for a raster coloured by net
        self._layer_nets: Dict[str, List[str]] = {}  # layer -> the names its shapes' net indices stand for
        self._styled: Dict[str, tuple] = {}  # layer -> the colouring its raster was drawn with
        self._net_ids: Dict[str, np.ndarray] = {}  # layer -> the net at each raster pixel (net_id_map)
        self._colors_cache: Dict[Tuple[str, tuple], Optional[List[QColor]]] = {}
        self._frame: Optional[_Frame] = None
        self._frame_busy = False  # a frame is being drawn; only one at a time
        self._frame_queued = False
        self._holes = np.zeros((0, 3))  # every hole, (x, y, diameter) mm: set_holes
        self._port_items: Dict[int, PortItem] = {}
        self._pair_items: List[PairItem] = []
        self._board_rect: Optional[QRectF] = None
        self._zooming = False
        self._rough = False  # a raster was drawn with fast scaling while zooming
        self._vector_ms = 0.0  # what the last full repaint of vector copper took
        self._panning = False
        self._pan_origin = None
        self._right_press: Optional[QPointF] = None
        self._fitted = False
        self._inflight: Set[str] = set()  # background jobs in flight, e.g. "raster:F_Cu"
        self._rendering = False
        self._shown: Dict[str, Tuple[str, QImage]] = {}  # layer -> (colour, coverage mask) of its raster
        self._marker: Optional[SnapMarker] = None
        self._edge_line: Optional[QGraphicsLineItem] = None
        self._normal_line: Optional[QGraphicsLineItem] = None
        self._rubber: Optional[QGraphicsLineItem] = None
        self._hover_edge: Optional[EdgeHit] = None
        self._pending_pos: Optional[Tuple[float, float, int, str]] = None
        self._pair_first: Optional[int] = None  # the pair tool's + port, once clicked
        self._probe_at: Optional[Tuple[float, float]] = None
        self.raster_loader = None  # callable(layer_name, px_per_mm) -> (width, height, coverage bytes)
        self.shape_loader = None  # callable(layer_name) -> gerber_shapes arrays
        self.copper = CopperTool(self)
        self.copper.snapChanged.connect(self.snapChanged)

        # One thread per layer in flight; they mostly wait on worker processes.
        self._background = BackgroundRunner(self, threads=4)
        self._smooth_timer = QTimer(self)
        self._smooth_timer.setSingleShot(True)
        self._smooth_timer.setInterval(SMOOTH_AFTER_ZOOM_MS)
        self._smooth_timer.timeout.connect(self._zoom_settled)
        self._probe_timer = QTimer(self)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(NET_PROBE_MS)
        self._probe_timer.timeout.connect(self._probe_net)
        theme.manager().changed.connect(self._theme_changed)  # bound: dropped when the canvas is deleted

    # ------------------------------------------------------------ mapping

    def zoom(self) -> float:
        return self.transform().m11()

    def to_mm(self, scene_point: QPointF):
        return scene_point.x() / self.base_ppm, self.height_mm - scene_point.y() / self.base_ppm

    def to_scene(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(x_mm * self.base_ppm, (self.height_mm - y_mm) * self.base_ppm)

    def _mm_to_scene(self) -> QTransform:
        ppm = self.base_ppm
        return QTransform(ppm, 0, 0, -ppm, 0, self.height_mm * ppm)

    def _scene_rect_mm(self, rect: QRectF) -> QRectF:
        ppm = self.base_ppm
        return QRectF(rect.left() / ppm, self.height_mm - rect.bottom() / ppm, rect.width() / ppm, rect.height() / ppm)

    def layer_names(self) -> List[str]:
        return self.snaps.layer_names

    def visible_sim_layers(self) -> List[int]:
        out = []
        for layer in (self.board or {}).get("layers", []):
            if layer.get("sim_index") is not None and self.visible_layers.get(layer["name"], True):
                out.append(int(layer["sim_index"]))
        return out

    def default_layer(self) -> int:
        visible = self.visible_sim_layers()
        return min(visible) if visible else 0

    def _radius_mm(self) -> float:
        return SNAP_RADIUS_PX / max(self.base_ppm * self.zoom(), 1e-9)

    # ----------------------------------------------------------- snapping

    def snap_point(self, x_mm: float, y_mm: float) -> Optional[SnapHit]:
        if not self.snapping or _no_snap_modifier() or not len(self.snaps):
            return None
        return self.snaps.nearest_point(x_mm, y_mm, self._radius_mm(), self.visible_sim_layers())

    def snap_scene(self, scene_point: QPointF) -> Tuple[QPointF, Optional[SnapHit]]:
        """A scene point moved onto the nearest snap target, and the target (None when free)."""
        hit = self.snap_point(*self.to_mm(scene_point))
        if hit is None:
            self.hide_snap()
            return scene_point, None
        snapped = self.to_scene(hit.x_mm, hit.y_mm)
        self._show_marker(snapped, hit.kind)
        self.snapChanged.emit(hit.label(self.layer_names()))
        return snapped, hit

    def _ensure_overlay(self) -> None:
        scene = self.scene()
        if self._marker is None:
            self._marker = SnapMarker()
            self._marker.setVisible(False)
            scene.addItem(self._marker)
        if self._edge_line is None:
            self._edge_line = scene.addLine(0, 0, 0, 0)
            self._edge_line.setZValue(55)
            self._edge_line.setVisible(False)
            self._normal_line = scene.addLine(0, 0, 0, 0)
            self._normal_line.setZValue(55)
            self._normal_line.setVisible(False)
            self._rubber = scene.addLine(0, 0, 0, 0)
            self._rubber.setZValue(55)
            self._rubber.setVisible(False)

    def show_snap(self, hit: SnapHit) -> None:
        """Mark a snap target that something dragged has landed on."""
        self._show_marker(self.to_scene(hit.x_mm, hit.y_mm), hit.kind)

    def _show_marker(self, scene_point: QPointF, kind: str) -> None:
        self._ensure_overlay()
        self._marker.kind = kind
        self._marker.setPos(scene_point)
        self._marker.setVisible(True)
        self._marker.update()

    def hide_snap(self) -> None:
        if self._marker is not None:
            self._marker.setVisible(False)
        if self._edge_line is not None:
            self._edge_line.setVisible(False)
            self._normal_line.setVisible(False)
        self._hover_edge = None
        self.snapChanged.emit("")

    def _show_edge(self, edge: EdgeHit) -> None:
        self._ensure_overlay()
        accent = QColor(theme.colors()["accent"])
        pen = QPen(accent, 3)
        pen.setCosmetic(True)
        p1, p2 = self.to_scene(edge.x1, edge.y1), self.to_scene(edge.x2, edge.y2)
        self._edge_line.setPen(pen)
        self._edge_line.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        self._edge_line.setVisible(True)
        # A short tick into the copper, so the side the port faces is visible.
        mx, my = edge.mid
        tick = 18.0 / max(self.base_ppm * self.zoom(), 1e-9)
        m, n = self.to_scene(mx, my), self.to_scene(mx + edge.nx * tick, my + edge.ny * tick)
        tick_pen = QPen(accent, 1.5, Qt.DashLine)
        tick_pen.setCosmetic(True)
        self._normal_line.setPen(tick_pen)
        self._normal_line.setLine(m.x(), m.y(), n.x(), n.y())
        self._normal_line.setVisible(True)
        self._show_marker(m, "edge_mid")
        self._hover_edge = edge

    # ------------------------------------------------------------ content

    def set_board(self, board: Dict, drills: List[Dict]) -> None:
        """A new board: rebuild everything and re-render the rasters."""
        self.board = board
        scene = self.scene()
        scene.clear()
        self.copper.reset()
        for channel in self._inflight:
            self._background.cancel(channel)
        # The files behind the rasters and shapes may have changed with the board.
        self._inflight.clear()
        self._shown.clear()
        self._update_rendering()
        self._layers.clear()
        self._shapes.clear()
        self._raw_shapes.clear()
        self._layer_nets.clear()
        self._styled.clear()
        self._net_ids.clear()
        self._colors_cache.clear()
        self._background.cancel("frame")  # drop any frame of the old board still drawing
        self._frame, self._frame_busy = None, False
        self._vector_ms = 0.0
        self._holes = np.zeros((0, 3))
        self._port_items.clear()
        self._pair_items = []
        self._board_rect = None
        self._marker = self._edge_line = self._normal_line = self._rubber = None
        self._pending_pos = None
        self._pair_first = None
        self._net_hex = palette.net_colors(net["name"] for net in board.get("nets", []))
        if self.highlight not in self._net_hex:
            self.highlight = None
        if not board.get("ready"):
            self._fitted = False
            self._redraw_board()
            return
        preview = board["preview"]
        self.base_ppm = float(preview["px_per_mm"])
        self.height_mm = float(board["board"]["height_mm"])
        width_px, height_px = float(preview["width_px"]), float(preview["height_px"])
        self._board_rect = QRectF(0, 0, width_px, height_px)
        scene.setSceneRect(QRectF(-width_px * 0.25, -height_px * 0.25, width_px * 1.5, height_px * 1.5))

        for layer in sorted(board.get("layers", []), key=lambda layer: -layer["index"]):
            self._layers[layer["name"]] = None
        self._layers["edge_cuts"] = None

        self.set_drill_holes(drills, redraw=False)
        self._redraw_board()
        for name in self._layers:
            self._render(name)
        for name in self._layers:  # after every raster: they are what shows first
            self._load_shapes(name)
        if not self._fitted:
            QTimer.singleShot(0, self, self.fit)
            self._fitted = True

    def set_drill_holes(self, drills: List[Dict], redraw: bool = True) -> None:
        """The drill file's holes, drawn with the board until the project has vias of its own."""
        self.set_holes([(h["x_mm"], h["y_mm"], h["diameter_mm"]) for h in drills], redraw)

    def set_holes(self, holes, redraw: bool = True) -> None:
        """Every hole on the board as (x, y, diameter) in mm."""
        self._holes = np.array(list(holes), float).reshape(-1, 3)
        if redraw:
            self._redraw_board()

    def _paint_holes(self, painter: QPainter, area: QRectF) -> None:
        """The holes in `area` (scene units); tiny ones are drawn as dots."""
        holes = self._holes
        if not len(holes) or area.isEmpty():
            return
        mm = self._scene_rect_mm(area)
        r = holes[:, 2] / 2
        inside = ((holes[:, 0] + r >= mm.left()) & (holes[:, 0] - r <= mm.right()) &
                  (holes[:, 1] + r >= mm.top()) & (holes[:, 1] - r <= mm.bottom()))
        c = theme.colors()
        painter.setPen(QPen(QColor(c["text_faint"]), 0))
        painter.setBrush(QColor(c["sunken"]))
        ppm, h = self.base_ppm, self.height_mm
        for x, y, d in holes[inside].tolist():
            painter.drawEllipse(QPointF(x * ppm, (h - y) * ppm), d / 2 * ppm, d / 2 * ppm)

    def has_nets(self) -> bool:
        return bool(self._net_hex)

    def net_color(self, net: str) -> str:
        return self._net_hex.get(net) or palette.no_net_color()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 - Qt override
        """The board: backing, copper bottom-up, outline, holes. Qt caches what this draws."""
        super().drawBackground(painter, rect)
        if self._board_rect is None:
            return
        painter.save()
        try:
            self._paint_board(painter, rect)
        finally:
            painter.restore()

    def _paint_board(self, painter: QPainter, exposed: QRectF) -> None:
        c = theme.colors()
        painter.setRenderHint(QPainter.Antialiasing)
        # Fast scaling while zooming; _zoom_settled smooths it.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, not self._zooming)
        painter.setPen(QPen(QColor(c["line_strong"]), 0))
        painter.setBrush(QColor(c["sunken"]))
        painter.drawRect(self._board_rect)
        world = painter.worldTransform()
        dpr = painter.device().devicePixelRatioF()
        scale = world.m11() * dpr
        stack = self._vector_stack() if scale > VECTOR_ABOVE else None
        # Only what is on screen.
        visible = self.mapToScene(self.viewport().rect()).boundingRect() & self._board_rect
        area = exposed & visible
        if stack and self._vector_ms <= INTERACTIVE_FRAME_MS:
            # Zoomed well in: draw straight from the shapes.
            if not area.isEmpty():
                started = time.perf_counter()
                layers = [(sync, all_dark, QColor(self._layer_color(name)), self._opacity(name),
                           self._net_colors(name)) for name, (sync, _own, all_dark) in stack]
                pixels = self._device_rect(world, dpr, area)
                image = render_stack(layers, pixels.size(), self._mm_to_pixels(world, dpr, pixels),
                                     self._scene_rect_mm(area))
                self._draw_aligned(painter, image, QPointF(pixels.topLeft()), dpr)
                if area.width() * area.height() * 2 >= visible.width() * visible.height():  # not a pan strip
                    self._vector_ms = (time.perf_counter() - started) * 1000
        elif stack:
            # Dense board zoomed out: use the last background frame, rasters around it, and request a new frame.
            frame = self._frame if self._frame is not None and self._frame.key == self._stack_key(stack) else None
            same_scale = frame is not None and abs(frame.scale - scale) <= 1e-9 * scale
            self._paint_rasters(painter, _outside(exposed, frame.rect) if frame is not None else None)
            if same_scale:
                corner = world.map(frame.rect.topLeft()) * dpr  # whole pixels: frames are drawn aligned
                self._draw_aligned(painter, frame.image, QPointF(round(corner.x()), round(corner.y())), dpr)
            elif frame is not None:  # the wheel is turning: stretched until the next frame
                painter.drawImage(frame.rect, frame.image)
                self._rough = self._rough or self._zooming
            covered = same_scale and frame.rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6).contains(visible)
            if not covered and not self._frame_queued:
                self._frame_queued = True
                QTimer.singleShot(0, self, self._request_frame)  # not from inside a paint; dropped with the canvas
        else:
            self._paint_rasters(painter, None)
        if self.show_drills:
            self._paint_holes(painter, exposed.adjusted(-1, -1, 1, 1))

    def _paint_rasters(self, painter: QPainter, rects: Optional[List[QRectF]]) -> None:
        """The visible layers' rasters, whole or only over `rects`."""
        for name, pixmap in self._layers.items():
            if pixmap is None or not self.visible_layers.get(name, True):
                continue
            whole = QRectF(pixmap.rect())
            for rect in [whole] if rects is None else [r & whole for r in rects]:
                if not rect.isEmpty():
                    painter.drawPixmap(rect, pixmap, rect)
                    self._rough = self._rough or self._zooming

    def _vector_stack(self) -> Optional[List[Tuple[str, Tuple[List[Shape], List[Shape], bool]]]]:
        """The visible layers' shapes, bottom first, or None while any of them has not loaded."""
        out = []
        for name in self._layers:
            if self.visible_layers.get(name, True):
                shapes = self._shapes.get(name)
                if shapes is None:
                    return None
                out.append((name, shapes))
        return out

    def _stack_key(self, stack) -> tuple:
        return tuple((name, self._layer_color(name)) for name, _shapes in stack) + (self._style_key(),)

    @staticmethod
    def _device_rect(world: QTransform, dpr: float, scene_rect: QRectF):
        """The whole device pixels covering a scene rect."""
        logical = world.mapRect(scene_rect)
        return QRectF(logical.x() * dpr, logical.y() * dpr, logical.width() * dpr, logical.height() * dpr).toAlignedRect()

    def _mm_to_pixels(self, world: QTransform, dpr: float, pixels) -> QTransform:
        """Board mm to the pixels of an image covering device rect `pixels`."""
        return self._mm_to_scene() * world * QTransform.fromScale(dpr, dpr) * \
            QTransform.fromTranslate(-pixels.x(), -pixels.y())

    @staticmethod
    def _draw_aligned(painter: QPainter, image: QImage, corner: QPointF, dpr: float) -> None:
        """An image of device pixels laid pixel for pixel with its top left on device point `corner`."""
        image.setDevicePixelRatio(dpr)
        painter.save()
        painter.setWorldTransform(QTransform())
        painter.drawImage(corner / dpr, image)
        painter.restore()

    def _request_frame(self) -> None:
        """Draw the view's copper from the shapes off the UI thread, unless it is already on its way."""
        self._frame_queued = False
        if self._frame_busy or self._board_rect is None or self._zooming or not self._sharp():
            return  # repaints when done, which asks again
        stack = self._vector_stack()
        visible = self.mapToScene(self.viewport().rect()).boundingRect() & self._board_rect
        if stack is None or visible.isEmpty():
            return
        world, dpr = self.viewportTransform(), self.devicePixelRatioF()
        pixels = self._device_rect(world, dpr, visible)
        scene_rect = world.inverted()[0].mapRect(QRectF(pixels.x() / dpr, pixels.y() / dpr,
                                                        pixels.width() / dpr, pixels.height() / dpr))
        key = self._stack_key(stack)
        layers = [(own, all_dark, QColor(self._layer_color(name)), self._opacity(name), self._net_colors(name))
                  for name, (_sync, own, all_dark) in stack]
        transform, visible_mm, size = self._mm_to_pixels(world, dpr, pixels), self._scene_rect_mm(scene_rect), pixels.size()
        self._frame_busy = True

        def work():
            started = time.perf_counter()
            image = render_stack(layers, size, transform, visible_mm)
            return image, (time.perf_counter() - started) * 1000

        def done(result) -> None:
            self._frame_busy = False
            image, ms = result
            self._vector_ms = ms  # cheap again, so the next paint draws in place
            self._frame = _Frame(scene_rect, world.m11() * dpr, key, image)
            self._redraw_board()

        def failed(_exc) -> None:
            self._frame_busy = False

        self._background.submit("frame", work, done, failed)

    def _sharp(self) -> bool:
        """Zoomed in past the rasters' own pixels, where layers are drawn from their shapes."""
        return self.zoom() * self.devicePixelRatioF() > VECTOR_ABOVE

    def _redraw_board(self) -> None:
        """Something under the ports changed: drop the cached background and repaint it."""
        self.resetCachedContent()
        self.viewport().update()

    def reload_layers(self, names: List[str]) -> None:
        """Re-render these layers after their Gerbers were rewritten."""
        for name in names:
            if name not in self._layers:
                continue
            for channel in (f"raster:{name}", f"shapes:{name}", f"nets:{name}", f"styled:{name}"):
                if channel in self._inflight:
                    self._background.cancel(channel)
                    self._inflight.discard(channel)
            self._shapes.pop(name, None)
            self._raw_shapes.pop(name, None)
            self._net_ids.pop(name, None)
            self._styled.pop(name, None)
            self._colors_cache = {k: v for k, v in self._colors_cache.items() if k[0] != name}
            self._render(name)
            self._load_shapes(name)
        self._frame = None
        self._vector_ms = 0.0
        self._update_rendering()
        self._redraw_board()

    def _zoom_settled(self) -> None:
        self._zooming = False
        if self._rough:  # only a fast-scaled raster needs redoing
            self._rough = False
            self._redraw_board()

    def set_snaps(self, snaps: SnapSet) -> None:
        self.snaps = snaps
        for item in self._port_items.values():
            item.prepareGeometryChange()  # the labels name the layers
            item.update()

    def set_references(self, references: Dict) -> None:
        self.references = references
        for item in self._port_items.values():
            item.prepareGeometryChange()
            item.update()

    def set_ports(self, ports, selected: Optional[int], pairs=(), selected_pair: Optional[int] = None) -> None:
        scene = self.scene()
        live = set()
        roles: Dict[int, Tuple[int, str]] = {}
        for i, pair in enumerate(pairs):
            if pair.positive is not None:
                roles[pair.positive] = (i, "+")
            if pair.negative is not None:
                roles.setdefault(pair.negative, (i, "−"))
        for port in ports:
            live.add(port.number)
            item = self._port_items.get(port.number)
            if item is None:
                item = PortItem(self, port)
                scene.addItem(item)
                self._port_items[port.number] = item
            item.prepareGeometryChange()
            item.port = port
            item.selected_port = port.number == selected
            item.pair = roles.get(port.number)
            item.pair_selected = item.pair is not None and item.pair[0] == selected_pair
            if not item._dragging:
                item.setPos(self.to_scene(port.x_mm, port.y_mm))
                item.setRotation(0 if item.is_circuit() else -port.rotation_deg)
            item.neg_handle.setVisible(item.is_circuit())
            if item.is_circuit() and not item.neg_handle._dragging:
                item.neg_handle.setPos(item.neg_local())
            item.update()
        for number in [n for n in self._port_items if n not in live]:
            scene.removeItem(self._port_items.pop(number))
        self._set_pairs(list(pairs), selected_pair)

    def _set_pairs(self, pairs, selected: Optional[int]) -> None:
        scene = self.scene()
        while len(self._pair_items) > len(pairs):
            scene.removeItem(self._pair_items.pop())
        for i, pair in enumerate(pairs):
            if i >= len(self._pair_items):
                item = PairItem(self, i, pair)
                scene.addItem(item)
                self._pair_items.append(item)
            item = self._pair_items[i]
            item.prepareGeometryChange()
            item.pair = pair
            item.selected_pair = i == selected
            item.setVisible(item._ports() is not None)
            item.update()

    def partner_item(self, number: int) -> Optional[PortItem]:
        """The other port of the pair port `number` is in."""
        item = self._port_items.get(number)
        if item is None or item.pair is None or not 0 <= item.pair[0] < len(self._pair_items):
            return None
        pair = self._pair_items[item.pair[0]].pair
        other = pair.negative if pair.positive == number else pair.positive
        return self._port_items.get(other) if other is not None else None

    def update_pairs(self) -> None:
        """A port moved: the brackets that join it follow."""
        for item in self._pair_items:
            item.prepareGeometryChange()
            item.update()

    def apply_layer_visibility(self) -> None:
        self._redraw_board()

    def set_drills_visible(self, visible: bool) -> None:
        self.show_drills = visible
        self.copper.show_holes = visible
        self.copper.refresh()
        self._redraw_board()

    def fit(self) -> None:
        if self._board_rect is None:
            return
        # Room round the board for edge-port labels and pair brackets.
        margin = max(20.0, 0.06 * max(self._board_rect.width(), self._board_rect.height()))
        self.fitInView(self._board_rect.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)
        self._geometry_changed()

    def _geometry_changed(self) -> None:
        """The zoom changed: ports and pairs are drawn at a constant size on screen."""
        for item in self._port_items.values():
            item.prepareGeometryChange()
            item.neg_handle.prepareGeometryChange()
        for item in self._pair_items:
            item.prepareGeometryChange()

    def focus_port(self, number: int) -> None:
        """Centre the view on a port, zooming in if the board is shown too small to see it."""
        item = self._port_items.get(number)
        if item is None:
            return
        if self.zoom() < 1.0:
            factor = 1.0 / self.zoom()
            self.scale(factor, factor)
            self._geometry_changed()
        self.centerOn(item)

    def focus_pair(self, index: int) -> None:
        if 0 <= index < len(self._pair_items):
            ports = self._pair_items[index]._ports()
            if ports is not None:
                self.focus_port(ports[0].number)
                self.centerOn((ports[0].pos() + ports[1].pos()) / 2)

    def focus_net(self, net: str) -> bool:
        """Fit the view to a net's copper on the shown layers; False when none of it is loaded or shown."""
        bounds: Optional[QRectF] = None
        for name, (sync, _own, _all_dark) in self._shapes.items():
            names = self._layer_nets.get(name, [])
            if net not in names or not self.visible_layers.get(name, True):
                continue
            index = names.index(net)
            for shape in sync:
                if shape.net == index and shape.dark:
                    bounds = shape.bounds if bounds is None else bounds.united(shape.bounds)
        if bounds is None:
            return False
        rect = QRectF(self.to_scene(bounds.left(), bounds.bottom()), self.to_scene(bounds.right(), bounds.top()))
        margin = max(rect.width(), rect.height()) * 0.2 + 2.0
        self.fitInView(rect.normalized().adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)
        if self.zoom() > MAX_ZOOM:
            self.scale(MAX_ZOOM / self.zoom(), MAX_ZOOM / self.zoom())
        self._geometry_changed()
        return True

    def port_at(self, pos: QPoint) -> int:
        """The number of the port drawn at a view position, or -1."""
        for item in self.items(pos):
            if isinstance(item, TerminalHandle):
                item = item.port_item
            if isinstance(item, PortItem):
                return item.number
        return -1

    def pair_at(self, pos: QPoint) -> int:
        item = self.itemAt(pos)
        return item.index if isinstance(item, PairItem) else -1

    # ------------------------------------------------------------ colour

    def set_color_by(self, mode: str) -> None:
        mode = "net" if mode == "net" else "layer"
        if mode != self.color_by:
            self.color_by = mode
            self._restyle()

    def set_highlight(self, net: Optional[str]) -> None:
        net = net if net in self._net_hex else None
        if net != self.highlight:
            self.highlight = net
            self._restyle()

    def _styled_mode(self) -> bool:
        return self.color_by == "net" or self.highlight is not None

    def _style_key(self) -> tuple:
        return (self.color_by, self.highlight, theme.manager().name)

    def _net_colors(self, name: str) -> Optional[List[QColor]]:
        """Colour per net index of layer `name` (last entry: no net), or None for a single-colour layer."""
        if not self._styled_mode() or name == "edge_cuts":
            return None
        key = (name, self._style_key())
        if key in self._colors_cache:
            return self._colors_cache[key]
        layer_color = QColor(self._layer_color(name))
        backing = QColor(theme.colors()["sunken"])
        out = []
        for net in self._layer_nets.get(name, []) + [None]:
            if self.color_by == "net":
                color = QColor(self.net_color(net) if net else palette.no_net_color())
            else:
                color = QColor(layer_color)
            if self.highlight is not None and net != self.highlight:
                color = palette.dimmed(color, backing)
            out.append(color)
        self._colors_cache[key] = out
        return out

    def _restyle(self) -> None:
        """The colouring changed: each layer's raster is drawn again to match, and frames too."""
        self._colors_cache.clear()
        self._frame = None
        for name in self._layers:
            if name == "edge_cuts":
                continue
            if self._styled_mode():
                self._render_styled(name)
                continue
            self._styled.pop(name, None)
            if name in self._shown:
                _color, mask = self._shown[name]
                current = self._layer_color(name)
                self._layers[name] = QPixmap.fromImage(self._tint(name, mask, current))
                self._shown[name] = (current, mask)
        self._redraw_board()

    def _coverage(self, name: str) -> Optional[np.ndarray]:
        shown = self._shown.get(name)
        if shown is None:
            return None
        mask = shown[1]
        height, width = mask.height(), mask.width()
        return np.frombuffer(mask.constBits(), np.uint8).reshape(height, mask.bytesPerLine())[:, :width].copy()

    def _build_net_ids(self, name: str) -> None:
        """Build a layer's per-pixel net map off the UI thread once it has its raster and shapes."""
        raw = self._raw_shapes.get(name)
        if name == "edge_cuts" or raw is None or name not in self._shown or name in self._net_ids:
            return
        channel = f"nets:{name}"
        if channel in self._inflight:
            return
        mask = self._shown[name][1]
        size = QSize(mask.width(), mask.height())
        width_mm = float(((self.board or {}).get("board") or {}).get("width_mm") or 0.0)
        if width_mm <= 0 or self.height_mm <= 0:
            return
        # The raster's size is rounded, so its scale differs slightly from the base one.
        sx, sy = size.width() / width_mm, size.height() / self.height_mm
        transform = QTransform(sx, 0, 0, -sy, 0, self.height_mm * sy)
        self._inflight.add(channel)
        self._update_rendering()

        def work():
            return net_id_map(decode_shapes(raw), size, transform)  # its own paths: no other thread has them

        def done(ids: np.ndarray) -> None:
            self._finished(channel)
            if name in self._layers:
                self._net_ids[name] = ids
                if self._styled_mode():
                    self._render_styled(name)

        self._background.submit(channel, work, done, lambda _exc: self._finished(channel))

    def _render_styled(self, name: str) -> None:
        """Recolour layer `name`'s raster by net, off the UI thread."""
        ids = self._net_ids.get(name)
        coverage = self._coverage(name)
        if ids is None or coverage is None or coverage.shape != ids.shape:
            self._build_net_ids(name)
            return
        key = self._style_key()
        if self._styled.get(name) == key:
            return
        colors = self._net_colors(name) or [QColor(self._layer_color(name))]
        fallback = QColor(palette.no_net_color()) if self.color_by == "net" else QColor(self._layer_color(name))
        if self.highlight is not None:
            fallback = palette.dimmed(fallback)
        opacity = self._opacity(name)
        channel = f"styled:{name}"
        self._inflight.add(channel)
        self._update_rendering()

        def done(image: QImage) -> None:
            self._finished(channel)
            if name in self._layers and self._style_key() == key:
                self._layers[name] = QPixmap.fromImage(image)
                self._styled[name] = key
                self._redraw_board()

        self._background.submit(channel, lambda: colorize(ids, coverage, colors, fallback, opacity), done,
                                lambda _exc: self._finished(channel))

    def net_at(self, x_mm: float, y_mm: float) -> Optional[Tuple[str, str]]:
        """(net, layer) of the topmost visible copper at a point; ("", layer) for copper on no net."""
        point = QPointF(x_mm, y_mm)
        for name in reversed(list(self._layers)):
            if name == "edge_cuts" or not self.visible_layers.get(name, True):
                continue
            loaded = self._shapes.get(name)
            if loaded is None:
                continue
            names = self._layer_nets.get(name, [])
            for shape in reversed(loaded[0]):
                if shape.bounds.contains(point) and shape.path.contains(point):
                    if not shape.dark:
                        break  # cleared here: nothing of this layer under the point
                    return (names[shape.net] if 0 <= shape.net < len(names) else ""), name
        return None

    def _probe_net(self) -> None:
        if self._probe_at is None or not self._net_hex:
            return
        hit = self.net_at(*self._probe_at)
        if hit is None:
            self.netHovered.emit("")
        else:
            net, layer = hit
            self.netHovered.emit(f"{net or 'no net'} · {layer}")

    # ------------------------------------------------------------ rasters

    def _layer_color(self, name: str) -> str:
        if name == "edge_cuts":
            return theme.colors()["text_dim"]
        for layer in (self.board or {}).get("layers", []):
            if layer["name"] == name:
                return layer["color"]
        return "#888888"

    @staticmethod
    def _opacity(name: str) -> float:
        return 1.0 if name == "edge_cuts" else COPPER_OPACITY

    @staticmethod
    def _tint(name: str, mask: QImage, color: str) -> QImage:
        return tinted(mask, color, BoardCanvas._opacity(name))

    def _render(self, name: str) -> None:
        """Rasterise layer `name` at base resolution, off the UI thread."""
        if self.raster_loader is None:
            return
        loader, ppm, color = self.raster_loader, self.base_ppm, self._layer_color(name)
        channel = f"raster:{name}"
        self._inflight.add(channel)
        self._update_rendering()

        def work():
            width, height, coverage = loader(name, ppm)
            layer_mask = mask_image(width, height, coverage)
            return layer_mask, self._tint(name, layer_mask, color)

        def done(result) -> None:
            layer_mask, image = result
            self._finished(channel)
            current = self._layer_color(name)
            if current != color:  # the theme changed while it was rendering
                image = self._tint(name, layer_mask, current)
            self._show(name, layer_mask, image, current)

        self._background.submit(channel, work, done, lambda _exc: self._finished(channel))

    def _load_shapes(self, name: str) -> None:
        """Fetch and decode layer `name`'s vector shapes, off the UI thread."""
        if self.shape_loader is None:
            return
        loader = self.shape_loader
        channel = f"shapes:{name}"
        self._inflight.add(channel)
        self._update_rendering()

        def work():
            raw = loader(name)
            shapes = decode_shapes(raw)
            return raw, (shapes, decode_shapes(raw), all(shape.dark for shape in shapes))

        def done(result) -> None:
            self._finished(channel)
            raw, shapes = result
            if name in self._layers:
                self._shapes[name] = shapes
                self._raw_shapes[name] = raw
                self._layer_nets[name] = [str(n) for n in raw.get("nets", [])] if "nets" in raw else []
                self._colors_cache = {k: v for k, v in self._colors_cache.items() if k[0] != name}
                if self._styled_mode() and name != "edge_cuts":
                    self._render_styled(name)
                if self._sharp():
                    self._redraw_board()

        # Without shapes, fall back to the raster.
        self._background.submit(channel, work, done, lambda _exc: self._finished(channel))

    def _show(self, name: str, mask: QImage, image: QImage, color: str) -> None:
        if name not in self._layers:
            return
        self._shown[name] = (color, mask)
        if self._styled.get(name) == self._style_key():
            return  # a raster coloured by net is already in its place
        self._layers[name] = QPixmap.fromImage(image)
        self._redraw_board()
        if self._styled_mode() and name != "edge_cuts":
            self._render_styled(name)

    def _finished(self, channel: str) -> None:
        self._inflight.discard(channel)
        self._update_rendering()
        kind, _, name = channel.partition(":")
        if kind in ("raster", "shapes") and f"raster:{name}" not in self._inflight \
                and f"shapes:{name}" not in self._inflight:
            self.copper.layer_redrawn(name)

    def _update_rendering(self) -> None:
        busy = bool(self._inflight)
        if busy != self._rendering:
            self._rendering = busy
            self.renderingChanged.emit(busy)

    def _theme_changed(self, _name: str) -> None:
        self._net_hex = palette.net_colors(net["name"] for net in (self.board or {}).get("nets", []))
        self._retint()

    def _retint(self) -> None:
        if self._styled_mode():
            self._restyle()
            for name, (color, mask) in list(self._shown.items()):
                if name == "edge_cuts":
                    current = self._layer_color(name)
                    if current != color:
                        self._show(name, mask, self._tint(name, mask, current), current)
        else:
            for name, (color, mask) in list(self._shown.items()):
                current = self._layer_color(name)
                if current != color:
                    self._show(name, mask, self._tint(name, mask, current), current)
        self._redraw_board()  # the board colours follow the theme too
        for item in list(self._port_items.values()) + self._pair_items:
            item.update()

    # -------------------------------------------------------------- tools

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            tool = "select"
        self.tool = tool
        self._pending_pos = None
        self._pair_first = None
        self._set_tool_target(None)
        if self._rubber is not None:
            self._rubber.setVisible(False)
        self.hide_snap()
        self.setCursor(Qt.ArrowCursor if tool == "select" else Qt.CrossCursor)
        if tool == "pair":
            self.snapChanged.emit("click the + port")
        elif tool == "via":
            self.snapChanged.emit("click to place a via (Esc ends)")

    def set_placing(self, placing: bool) -> None:
        """Kept for callers of the old single tool."""
        self.set_tool("point" if placing else "select")

    def _finish_tool(self) -> None:
        self.set_tool("select")
        self.toolFinished.emit()

    def _set_tool_target(self, number: Optional[int]) -> None:
        if number == self.tool_target:
            return
        old, self.tool_target = self.tool_target, number
        for n in (old, number):
            item = self._port_items.get(n) if n is not None else None
            if item is not None:
                item.update()

    def _terminal(self, x_mm: float, y_mm: float, fallback_layer: int) -> Tuple[float, float, int, str]:
        hit = self.snap_point(x_mm, y_mm)
        if hit is None:
            return x_mm, y_mm, fallback_layer, ""
        layer = hit.layer if hit.layer >= 0 else fallback_layer
        return hit.x_mm, hit.y_mm, layer, hit.label(self.layer_names())

    def _hover(self, scene_point: QPointF, view_pos: Optional[QPoint] = None) -> None:
        x_mm, y_mm = self.to_mm(scene_point)
        if self.tool == "edge":
            edge = None
            if len(self.snaps) and not _no_snap_modifier():
                edge = self.snaps.nearest_edge(x_mm, y_mm, self._radius_mm(), self.visible_sim_layers())
            if edge is None:
                self.hide_snap()
                self.snapChanged.emit("hover a copper edge")
                return
            self._show_edge(edge)
            names = self.layer_names()
            layer = names[edge.layer] if 0 <= edge.layer < len(names) else "?"
            self.snapChanged.emit(f"edge {edge.length_mm * 1000:.0f} µm · {layer}")
            return
        if self.tool == "pair":
            number = self.port_at(view_pos) if view_pos is not None else -1
            target = number if number >= 0 and number != self._pair_first else None
            self._set_tool_target(target)
            if self._pair_first is not None:
                self._ensure_overlay()
                first = self._port_items.get(self._pair_first)
                if first is not None:
                    pen = QPen(palette.pair_color(len(self._pair_items)), 2.0, Qt.DashLine)
                    pen.setCosmetic(True)
                    self._rubber.setPen(pen)
                    end = self._port_items[target].pos() if target is not None else scene_point
                    self._rubber.setLine(first.pos().x(), first.pos().y(), end.x(), end.y())
                    self._rubber.setVisible(True)
            which = "−" if self._pair_first is not None else "+"
            self.snapChanged.emit(f"P{target + 1} as the {which} port" if target is not None
                                  else f"click the {which} port")
            return
        if self.tool in ("point", "circuit", "via"):
            snapped, _hit = self.snap_scene(scene_point)
            if self.tool == "circuit" and self._pending_pos is not None:
                self._ensure_overlay()
                start = self.to_scene(self._pending_pos[0], self._pending_pos[1])
                pen = QPen(QColor(theme.colors()["accent"]), 1.5, Qt.DashLine)
                pen.setCosmetic(True)
                self._rubber.setPen(pen)
                self._rubber.setLine(start.x(), start.y(), snapped.x(), snapped.y())
                self._rubber.setVisible(True)

    # ------------------------------------------------------------- input

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        factor = math.exp(event.angleDelta().y() * 0.0016)
        zoom = self.zoom()
        factor = max(MIN_ZOOM / zoom, min(MAX_ZOOM / zoom, factor))
        self._zooming = True
        self._smooth_timer.start()
        self.scale(factor, factor)
        self._geometry_changed()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        pan = event.button() in (Qt.MiddleButton, Qt.RightButton) or \
            (event.button() == Qt.LeftButton and (event.modifiers() & (Qt.ShiftModifier | Qt.AltModifier)))
        if pan:
            self._panning = True
            self._pan_origin = event.position()
            self._right_press = event.position() if event.button() == Qt.RightButton else None
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            ready = bool(self.board and self.board.get("ready"))
            scene_point = self.mapToScene(event.position().toPoint())
            x_mm, y_mm = self.to_mm(scene_point)
            if self.tool == "select":
                item = self.itemAt(event.position().toPoint())
                if isinstance(item, (PortItem, TerminalHandle, PairItem)):
                    super().mousePressEvent(event)
                    return
                if ready and self.copper.press(event.position(), event.modifiers()):
                    return
                self.emptyClicked.emit()
            elif ready and self.tool == "point":
                x, y, layer, label = self._terminal(x_mm, y_mm, -1)
                self.placeRequested.emit(x, y, layer, label)
                self._finish_tool()
                return
            elif ready and self.tool == "via":
                # Keep placing vias until Esc.
                x, y, _layer, label = self._terminal(x_mm, y_mm, -1)
                self.viaRequested.emit(x, y, label)
                return
            elif ready and self.tool == "edge":
                if self._hover_edge is None:
                    self._hover(scene_point)
                edge = self._hover_edge
                if edge is not None:
                    names = self.layer_names()
                    layer = names[edge.layer] if 0 <= edge.layer < len(names) else "?"
                    self.edgePortRequested.emit(edge, f"edge · {layer}")
                    self._finish_tool()
                return
            elif ready and self.tool == "circuit":
                if self._pending_pos is None:
                    self._pending_pos = self._terminal(x_mm, y_mm, self.default_layer())
                    self.snapChanged.emit("now click the − terminal (Esc cancels)")
                else:
                    pos = self._pending_pos
                    neg = self._terminal(x_mm, y_mm, pos[2])
                    self.circuitPortRequested.emit(pos[:3], neg[:3], pos[3], neg[3])
                    self._finish_tool()
                return
            elif ready and self.tool == "pair":
                number = self.port_at(event.position().toPoint())
                if number < 0 or number == self._pair_first:
                    self.snapChanged.emit("click a port" if self._pair_first is None else
                                          "now click the − port (Esc cancels)")
                elif self._pair_first is None:
                    self._pair_first = number
                    self._set_tool_target(None)
                    self.snapChanged.emit(f"P{number + 1} is +; now click the − port (Esc cancels)")
                else:
                    first = self._pair_first
                    self._finish_tool()
                    self.pairRequested.emit(first, number)
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton and self.tool == "select" and self._net_hex \
                and not isinstance(self.itemAt(event.position().toPoint()), (PortItem, TerminalHandle, PairItem)):
            hit = self.net_at(*self.to_mm(self.mapToScene(event.position().toPoint())))
            if hit is not None and hit[0]:
                self.netActivated.emit(hit[0])
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._panning and self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self.board and self.board.get("ready"):
            scene_point = self.mapToScene(event.position().toPoint())
            x_mm, y_mm = self.to_mm(scene_point)
            self.cursorMoved.emit(x_mm, y_mm)
            if self.tool != "select":
                self._hover(scene_point, event.position().toPoint())
            elif event.buttons() & Qt.LeftButton:
                if self.copper.move(event.position()):
                    return
            elif not event.buttons() and not isinstance(self.itemAt(event.position().toPoint()),
                                                        (PortItem, TerminalHandle, PairItem)):
                self.copper.hover_moved(event.position())
            if self._net_hex and not event.buttons():
                self._probe_at = (x_mm, y_mm)
                self._probe_timer.start()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor)
            press, self._right_press = self._right_press, None
            if press is not None and event.button() == Qt.RightButton and self.tool == "select":
                moved = event.position() - press
                if abs(moved.x()) <= CLICK_TOLERANCE_PX and abs(moved.y()) <= CLICK_TOLERANCE_PX:
                    pos = event.position().toPoint()
                    number = self.port_at(pos)
                    pair = self.pair_at(pos) if number < 0 else -1
                    where = event.globalPosition().toPoint()
                    ref = handle = point = None
                    if number < 0 and pair < 0 and self.board and self.board.get("ready"):
                        ref, handle, point = self.copper.item_at(event.position())
                    # Deferred: a menu's event loop must not run inside this release.
                    if pair >= 0:
                        QTimer.singleShot(0, self, lambda: self.pairMenuRequested.emit(pair, where))
                    elif ref is not None:
                        QTimer.singleShot(0, self, lambda: self.copperMenuRequested.emit(ref, handle, point, where))
                    else:
                        QTimer.singleShot(0, self, lambda: self.contextRequested.emit(number, where))
            return
        if event.button() == Qt.LeftButton and self.tool == "select" and self.copper.release(event.position()):
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key_Escape and self.tool != "select":
            self._finish_tool()
            return
        if event.key() == Qt.Key_Escape and self.copper.cancel():
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.hide_snap()
        self.copper.hover_left()
        self._probe_timer.stop()
        self._probe_at = None
        self.netHovered.emit("")
        super().leaveEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - a right click is handled on release
        event.accept()

    def shutdown(self) -> None:
        self._smooth_timer.stop()
        self._probe_timer.stop()
        self._background.shutdown()
