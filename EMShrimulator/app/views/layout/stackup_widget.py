"""To-scale stackup cross-section, with each copper band paired to its gerber by name."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from EMShrimulator.app import theme

COPPER_MIN_PX = 9
DIELECTRIC_MIN_PX = 20
MAX_TOTAL_PX = 380


class StackupWidget(QWidget):
    layerClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._stackup: List = []
        self._gerbers: Dict[str, Dict] = {}
        self._selected: Optional[int] = None
        self._bands: List[tuple] = []
        theme.manager().changed.connect(self._theme_changed)  # bound: dropped when the widget is deleted

    def _theme_changed(self, _name: str) -> None:
        self.update()

    def set_data(self, stackup: List, board_layers: List[Dict], selected: Optional[int]) -> None:
        self._stackup = list(stackup)
        self._gerbers = {layer["name"].lower(): layer for layer in board_layers}
        self._selected = selected
        self._layout_bands()
        self.updateGeometry()
        self.update()

    def summary(self) -> str:
        copper = [layer for layer in self._stackup if layer.type == "copper"]
        total = sum(max(layer.thickness_mm or 0.0, 0.0) for layer in self._stackup)
        names = {layer.name.replace(".", "_").lower() for layer in copper}
        unused = [g for key, g in self._gerbers.items() if key not in names]
        missing = [layer.name for layer in copper if layer.name.replace(".", "_").lower() not in self._gerbers]
        parts = [f"{total:.3f} mm total" if total else "no thickness", f"{len(copper)} copper",
                 f"{len(self._gerbers)} copper gerber(s)"]
        text = " · ".join(parts)
        if unused:
            text += f"\n{len(unused)} copper gerber(s) have no stackup layer: drawn, never simulated."
        if missing:
            text += f"\n{', '.join(missing)} has no gerber: simulates as bare dielectric."
        return text

    def _layout_bands(self) -> None:
        total = sum(max(layer.thickness_mm or 0.0, 0.0) for layer in self._stackup)
        floors = sum(COPPER_MIN_PX if layer.type == "copper" else DIELECTRIC_MIN_PX for layer in self._stackup)
        scale = max((MAX_TOTAL_PX - floors) / total, 0.0) if total > 0 else 0.0
        y = 0.0
        self._bands = []
        for index, layer in enumerate(self._stackup):
            copper = layer.type == "copper"
            height = (COPPER_MIN_PX if copper else DIELECTRIC_MIN_PX) + max(layer.thickness_mm or 0.0, 0.0) * scale
            self._bands.append((index, y, height, layer))
            y += height

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        height = self._bands[-1][1] + self._bands[-1][2] if self._bands else 40
        return QSize(260, int(height) + 2)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c = theme.colors()
        width = self.width()
        if not self._bands:
            painter.setPen(QColor(c["text_dim"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "No stackup yet. Apply a preset, or add layers.")
            return
        small = QFont(self.font())
        small.setPointSizeF(max(small.pointSizeF() - 1, 6.5))
        for index, y, height, layer in self._bands:
            rect = QRectF(0, y, width - 1, height)
            copper = layer.type == "copper"
            gerber = self._gerbers.get(layer.name.replace(".", "_").lower()) if copper else None
            if copper:
                fill = QColor(gerber["color"]) if gerber else QColor(c["text_faint"])
            else:
                fill = QColor("#3f7f5a" if not theme.manager().is_light() else "#b9dcc6")
                fill.setAlpha(150)
            painter.fillRect(rect, fill)
            if copper and gerber is None:
                painter.setPen(QPen(QColor(c["danger"]), 1, Qt.DashLine))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            if index == self._selected:
                painter.setPen(QPen(QColor(c["accent"]), 2))
                painter.drawRect(rect.adjusted(1, 1, -1, -1))
            painter.setFont(small)
            painter.setPen(QColor("#0b0d11") if copper or theme.manager().is_light() else QColor(c["text"]))
            if height >= 11:
                detail = (gerber["file"].rsplit("-", 1)[-1] if gerber else "no gerber") if copper else \
                    " · ".join(v for v in (layer.material, f"εr {layer.epsilon:g}" if layer.epsilon else None) if v)
                thickness = f"{layer.thickness_mm:g} mm" if layer.thickness_mm else "—"
                text_rect = rect.adjusted(6, 0, -6, 0)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, layer.name)
                painter.drawText(text_rect.adjusted(width * 0.35, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, detail)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignRight, thickness)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        y = event.position().y()
        for index, top, height, _layer in self._bands:
            if top <= y < top + height:
                self.layerClicked.emit(index)
                return
