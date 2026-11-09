"""A colour swatch and opacity slider for a material or object."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


class AppearanceEdit(QWidget):
    """Commits a colour ("#rrggbb") and an opacity (0.05 to 1) when either changes; a slider drag commits on release."""

    colorCommitted = Signal(str)
    opacityCommitted = Signal(float)

    def __init__(self, dialog_title: str, parent=None):
        super().__init__(parent)
        self._title = dialog_title
        self._color = "#888888"
        self._opacity = 1.0
        self.swatch = QPushButton()
        self.swatch.setFixedWidth(64)
        self.swatch.setToolTip("Pick the colour")
        self.swatch.clicked.connect(self._pick)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 100)
        self.slider.setToolTip("Opacity in the 3D view")
        self.slider.sliderReleased.connect(self._commit_opacity)
        self.slider.valueChanged.connect(self._value_changed)
        self.percent = QLabel()
        self.percent.setMinimumWidth(36)
        self.percent.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.swatch)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.percent)

    def set_value(self, color: str, opacity: float) -> None:
        self._color, self._opacity = color, opacity
        self.swatch.setStyleSheet(f"background: {color}; border: 1px solid #00000040;")
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(opacity * 100)))
            self.slider.blockSignals(False)
            self.percent.setText(f"{self.slider.value()} %")

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, self._title)
        if chosen.isValid() and chosen.name() != QColor(self._color).name():
            self.colorCommitted.emit(chosen.name())

    def _value_changed(self, value: int) -> None:
        self.percent.setText(f"{value} %")
        if not self.slider.isSliderDown():
            self._commit_opacity()

    def _commit_opacity(self) -> None:
        value = self.slider.value() / 100
        if value != round(self._opacity, 2):
            self.opacityCommitted.emit(value)
