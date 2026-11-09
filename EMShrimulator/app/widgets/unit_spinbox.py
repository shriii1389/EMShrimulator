"""Numeric fields that emit `committed` once per finished edit, with an optional display scale."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


class UnitSpinBox(QDoubleSpinBox):
    committed = Signal(float)

    def __init__(self, unit: str = "", *, scale: float = 1.0, decimals: int = 4,
                 minimum: float = -1e12, maximum: float = 1e12, step: float = 1.0,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self._scale = scale
        self._last: Optional[float] = None
        self.setDecimals(decimals)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setAccelerated(True)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        if unit:
            self.setSuffix(f" {unit}")
        if tooltip:
            self.setToolTip(tooltip)
        self.editingFinished.connect(self._finish)
        self.valueChanged.connect(self._stepped)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        # Trim trailing zeros so 0.035 does not read as 0.0350.
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"

    def set_stored(self, stored: Optional[float]) -> None:
        """Show a stored value without emitting a commit."""
        if self.hasFocus():
            return
        shown = 0.0 if stored is None or not math.isfinite(stored) else stored / self._scale
        self._last = shown
        self.blockSignals(True)
        self.setValue(shown)
        self.blockSignals(False)

    def stored(self) -> float:
        return self.value() * self._scale

    def _finish(self) -> None:
        if self._last is None or abs(self.value() - self._last) > 1e-12:
            self._last = self.value()
            self.committed.emit(self.stored())

    def _stepped(self, _value: float) -> None:
        # Arrow keys and wheel steps commit immediately.
        if not self.lineEdit().hasFocus() or not self.lineEdit().isModified():
            self._finish()


class IntSpinBox(QSpinBox):
    committed = Signal(int)

    def __init__(self, *, minimum: int = -2**31 + 1, maximum: int = 2**31 - 1, step: int = 1,
                 suffix: str = "", tooltip: str = "", parent=None):
        super().__init__(parent)
        self._last: Optional[int] = None
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        if suffix:
            self.setSuffix(f" {suffix}")
        if tooltip:
            self.setToolTip(tooltip)
        self.editingFinished.connect(self._finish)
        self.valueChanged.connect(lambda _v: None if self.lineEdit().isModified() else self._finish())

    def set_stored(self, value: Optional[int]) -> None:
        if self.hasFocus():
            return
        self._last = int(value or 0)
        self.blockSignals(True)
        self.setValue(self._last)
        self.blockSignals(False)

    def _finish(self) -> None:
        if self._last is None or self.value() != self._last:
            self._last = self.value()
            self.committed.emit(self.value())
