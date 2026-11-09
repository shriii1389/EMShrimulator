"""Small line icons for the project tree, drawn with QPainter in the theme's colours and cached."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

SIZE = 16


def _pen(color: str, width: float = 1.3) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _draw(name: str, p: QPainter, color: str, accent: str) -> None:
    p.setPen(_pen(color))
    p.setBrush(Qt.NoBrush)
    if name == "project":
        p.drawRoundedRect(QRectF(2.5, 3.5, 11, 10), 1.5, 1.5)
        p.drawLine(QPointF(2.5, 6.5), QPointF(13.5, 6.5))
    elif name == "model":  # a cube
        p.drawPolygon(QPolygonF([QPointF(8, 2), QPointF(14, 5), QPointF(14, 11), QPointF(8, 14),
                                 QPointF(2, 11), QPointF(2, 5)]))
        p.drawPolyline(QPolygonF([QPointF(2, 5), QPointF(8, 8), QPointF(14, 5)]))
        p.drawLine(QPointF(8, 8), QPointF(8, 14))
    elif name == "nonmodel":  # the model's cube, dashed: drawn, never solved
        pen = _pen(color)
        pen.setDashPattern([1.6, 1.6])
        p.setPen(pen)
        p.drawPolygon(QPolygonF([QPointF(8, 2), QPointF(14, 5), QPointF(14, 11), QPointF(8, 14),
                                 QPointF(2, 11), QPointF(2, 5)]))
        p.drawPolyline(QPolygonF([QPointF(2, 5), QPointF(8, 8), QPointF(14, 5)]))
        p.drawLine(QPointF(8, 8), QPointF(8, 14))
    elif name == "solid":
        p.setBrush(QColor(color).darker(160))
        p.drawPolygon(QPolygonF([QPointF(8, 3), QPointF(13, 5.5), QPointF(13, 10.5), QPointF(8, 13),
                                 QPointF(3, 10.5), QPointF(3, 5.5)]))
    elif name == "sheet":
        p.drawPolygon(QPolygonF([QPointF(2, 10), QPointF(7, 5), QPointF(14, 5), QPointF(9, 10)]))
    elif name == "line":
        path = QPainterPath(QPointF(2, 12))
        path.cubicTo(QPointF(6, 2), QPointF(10, 14), QPointF(14, 4))
        p.drawPath(path)
    elif name == "cs":
        p.drawLine(QPointF(4, 12), QPointF(14, 12))
        p.drawLine(QPointF(4, 12), QPointF(4, 2))
        p.drawLine(QPointF(4, 12), QPointF(1.5, 14.5))
    elif name == "material":
        p.setBrush(QColor(color).darker(140))
        p.drawEllipse(QRectF(3, 3, 10, 10))
    elif name == "port":
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(accent))
        p.drawPolygon(QPolygonF([QPointF(9, 1.5), QPointF(3.5, 9), QPointF(7.5, 9), QPointF(6.5, 14.5),
                                 QPointF(12.5, 6.5), QPointF(8.5, 6.5)]))
    elif name in ("port_driven", "port_sense"):  # a layout port: its sheet and a triangle
        p.setPen(_pen(color, 2.2))
        p.drawLine(QPointF(3, 12.5), QPointF(13, 12.5))
        p.setPen(_pen(color, 1.2))
        p.setBrush(QColor(color) if name == "port_driven" else Qt.NoBrush)
        p.drawPolygon(QPolygonF([QPointF(8, 3), QPointF(4.5, 9.5), QPointF(11.5, 9.5)]))
    elif name == "pair":  # two traces joined by a bracket
        p.drawLine(QPointF(2.5, 5), QPointF(9, 5))
        p.drawLine(QPointF(2.5, 11), QPointF(9, 11))
        p.setPen(_pen(accent, 1.4))
        p.drawArc(QRectF(7.5, 5, 6, 6), -90 * 16, 180 * 16)
    elif name == "dump":
        p.drawRect(QRectF(2.5, 4.5, 11, 8))
        for x in (5.5, 8, 10.5):
            p.drawLine(QPointF(x, 6.5), QPointF(x, 10.5))
    elif name == "variable":
        font = p.font()
        font.setPixelSize(12)
        font.setItalic(True)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(0, 0, SIZE, SIZE), Qt.AlignCenter, "x")
    elif name == "setup":  # a gear: ring and teeth
        p.drawEllipse(QRectF(5, 5, 6, 6))
        for i in range(8):
            p.save()
            p.translate(8, 8)
            p.rotate(i * 45)
            p.drawLine(QPointF(0, 4.5), QPointF(0, 6.5))
            p.restore()
    elif name == "section":
        p.drawLine(QPointF(4, 5), QPointF(12, 5))
        p.drawLine(QPointF(4, 8), QPointF(12, 8))
        p.drawLine(QPointF(4, 11), QPointF(9, 11))
    elif name == "sweep":
        p.drawPolyline(QPolygonF([QPointF(2, 12), QPointF(5, 6), QPointF(8, 10), QPointF(11, 4), QPointF(14, 8)]))
    elif name == "results":
        p.drawLine(QPointF(2.5, 13.5), QPointF(13.5, 13.5))
        p.drawLine(QPointF(2.5, 13.5), QPointF(2.5, 2.5))
        p.drawPolyline(QPolygonF([QPointF(4, 11), QPointF(7, 6), QPointF(10, 9), QPointF(13, 3)]))
    elif name == "plot":
        p.drawRect(QRectF(2.5, 2.5, 11, 11))
        p.setPen(_pen(accent))
        p.drawPolyline(QPolygonF([QPointF(4, 11), QPointF(7, 6), QPointF(10, 9), QPointF(12, 5)]))
    elif name == "pattern":
        p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
        p.setPen(_pen(accent))
        p.drawEllipse(QRectF(5.5, 2.5, 5, 7))
    elif name == "log":
        p.drawRect(QRectF(2.5, 2.5, 11, 11))
        p.drawPolyline(QPolygonF([QPointF(4.5, 6), QPointF(6.5, 8), QPointF(4.5, 10)]))
        p.drawLine(QPointF(8, 10.5), QPointF(11.5, 10.5))
    elif name == "check":
        p.drawPolyline(QPolygonF([QPointF(3, 8.5), QPointF(6.5, 12), QPointF(13, 4)]))
    elif name == "warning":
        p.setPen(_pen(accent))
        p.drawPolygon(QPolygonF([QPointF(8, 2), QPointF(14.5, 13.5), QPointF(1.5, 13.5)]))
        p.drawLine(QPointF(8, 6), QPointF(8, 9.5))
        p.drawPoint(QPointF(8, 11.8))
    elif name == "runs":
        p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
        p.drawPolyline(QPolygonF([QPointF(8, 5), QPointF(8, 8), QPointF(10.5, 9.5)]))
    # The board's icons.
    elif name == "board":
        p.drawRoundedRect(QRectF(2, 3, 12, 10), 1.5, 1.5)
        p.setPen(_pen(accent))
        p.drawPolyline(QPolygonF([QPointF(4.5, 10.5), QPointF(7, 8), QPointF(10.5, 8)]))
        p.drawEllipse(QRectF(10.5, 6.8, 2.4, 2.4))
    elif name == "layers":  # a sheet with two more under it
        p.drawPolygon(QPolygonF([QPointF(2, 5.5), QPointF(8, 2.5), QPointF(14, 5.5), QPointF(8, 8.5)]))
        for y in (8.5, 11):
            p.drawPolyline(QPolygonF([QPointF(2, y), QPointF(8, y + 3), QPointF(14, y)]))
    elif name == "file":
        p.drawPolygon(QPolygonF([QPointF(4, 2.5), QPointF(9.5, 2.5), QPointF(12.5, 5.5), QPointF(12.5, 13.5),
                                 QPointF(4, 13.5)]))
        p.drawPolyline(QPolygonF([QPointF(9.5, 2.5), QPointF(9.5, 5.5), QPointF(12.5, 5.5)]))
    elif name == "stackup":  # a cross-section: copper, dielectric, copper
        p.setBrush(QColor(accent))
        p.drawRect(QRectF(2.5, 3, 11, 2))
        p.drawRect(QRectF(2.5, 11, 11, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(2.5, 5, 11, 6))
    elif name == "hole":
        p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
        p.setBrush(QColor(color))
        p.drawEllipse(QRectF(5.5, 5.5, 5, 5))
    elif name == "outline":
        pen = _pen(color)
        pen.setDashPattern([2.0, 1.6])
        p.setPen(pen)
        p.drawRoundedRect(QRectF(2.5, 3.5, 11, 9), 2, 2)
    # Snap targets, as the 3D view marks them.
    elif name in ("snap_face", "snap_edge", "snap_corner", "snap_grid"):
        p.setPen(_pen(accent, 1.8))
        if name == "snap_face":
            p.drawEllipse(QRectF(3, 3, 10, 10))
        elif name == "snap_edge":
            p.drawPolygon(QPolygonF([QPointF(8, 2.5), QPointF(14, 13), QPointF(2, 13)]))
        elif name == "snap_corner":
            p.drawRect(QRectF(3, 3, 10, 10))
        else:
            p.drawLine(QPointF(8, 2), QPointF(8, 14))
            p.drawLine(QPointF(2, 8), QPointF(14, 8))
    elif name == "measure":  # a ruler
        p.save()
        p.translate(8, 8)
        p.rotate(-35)
        p.drawRect(QRectF(-7, -2.5, 14, 5))
        for x in (-4.5, -2, 0.5, 3):
            p.drawLine(QPointF(x, -2.5), QPointF(x, -0.5 if x in (-2, 3) else 0.5))
        p.restore()
    else:
        p.drawEllipse(QRectF(5, 5, 6, 6))


@lru_cache(maxsize=256)
def _icon(name: str, color: str, accent: str, ratio: float) -> QIcon:
    pixmap = QPixmap(int(SIZE * ratio), int(SIZE * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    _draw(name, painter, color, accent)
    painter.end()
    return QIcon(pixmap)


def icon(name: str, color: str, accent: str = "", ratio: float = 2.0) -> QIcon:
    return _icon(name, color, accent or color, ratio)


@lru_cache(maxsize=256)
def swatch(color: str, ratio: float = 2.0) -> QIcon:
    """A filled rounded square - a material's colour."""
    pixmap = QPixmap(int(SIZE * ratio), int(SIZE * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(color).darker(150), 1))
    painter.setBrush(QBrush(QColor(color)))
    painter.drawRoundedRect(QRectF(3, 3, 10, 10), 2, 2)
    painter.end()
    return QIcon(pixmap)
