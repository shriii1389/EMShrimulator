"""Label/field rows, the hint paragraphs between them, and a few formatters."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(5)
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return form


def hint(text: str = "", role: str = "hint") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", role)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def button(text: str, *, role: str = "", tooltip: str = "", checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    if role:
        b.setProperty("role", role)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCheckable(checkable)
    return b


def card(title: str, about: str = "") -> Tuple[QFrame, QVBoxLayout]:
    """A titled settings card with a description; fill the returned layout."""
    frame = QFrame()
    frame.setProperty("role", "card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 12, 16, 14)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setProperty("role", "card-title")
    layout.addWidget(heading)
    if about:
        layout.addWidget(hint(about))
    return frame, layout


def centred_column(widgets: Iterable[QWidget], max_width: int = 860) -> QWidget:
    """Cards one under another, centred on a page no wider than reads well."""
    column = QVBoxLayout()
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(12)
    for widget in widgets:
        column.addWidget(widget)
    column.addStretch(1)
    inner = QWidget()
    inner.setLayout(column)
    inner.setMaximumWidth(max_width)
    host = QWidget()
    centred = QHBoxLayout(host)
    centred.setContentsMargins(20, 16, 20, 24)
    centred.addStretch(1)
    centred.addWidget(inner, 100)
    centred.addStretch(1)
    return host


def row(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for i, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and i == len(widgets) - 1) else 0)
    if not stretch_last:
        layout.addStretch(1)
    return host


class ChoiceBox(QComboBox):
    """A combo box over (value, label) pairs that commits the value, not the index."""

    committed = Signal(object)

    def __init__(self, options: Iterable[Tuple[object, str]] = (), tooltip: str = "", parent=None):
        super().__init__(parent)
        self.set_options(options)
        if tooltip:
            self.setToolTip(tooltip)
        self.activated.connect(lambda i: self.committed.emit(self.itemData(i)))

    def set_options(self, options: Iterable[Tuple[object, str]]) -> None:
        options = list(options)
        # Skip rebuilding an unchanged list.
        if options == getattr(self, "_options", None):
            return
        self._options = options
        current = self.currentData()
        self.blockSignals(True)
        self.clear()
        for value, label in options:
            self.addItem(label, value)
        self.blockSignals(False)
        if current is not None:
            self.set_value(current)

    def set_value(self, value: object) -> None:
        index = self.findData(value)
        self.blockSignals(True)
        if index >= 0:
            self.setCurrentIndex(index)
        self.blockSignals(False)

    def value(self) -> object:
        return self.currentData()


class Segmented(QWidget):
    """A row of mutually exclusive toggle buttons."""

    committed = Signal(object)

    def __init__(self, options: Sequence[Tuple[object, str]], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._values = []
        for i, (value, label) in enumerate(options):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumWidth(34)
            self._group.addButton(b, i)
            self._values.append(value)
            layout.addWidget(b)
        self._group.idClicked.connect(lambda i: self.committed.emit(self._values[i]))

    def set_value(self, value: object) -> None:
        if value in self._values:
            self._group.button(self._values.index(value)).setChecked(True)
            return
        # A value none of the buttons stands for: show none checked.
        self._group.setExclusive(False)
        for b in self._group.buttons():
            b.setChecked(False)
        self._group.setExclusive(True)

    def value(self) -> Optional[object]:
        checked = self._group.checkedId()
        return self._values[checked] if checked >= 0 else None

# --- formatting ---

_PREFIXES = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
             (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]


def fmt_eng(value: Optional[float], unit: str = "", digits: int = 3) -> str:
    """12.3 GHz, 4.56 ps, 789 µm - the engineering form of a number."""
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0:
        return f"0 {unit}".strip()
    magnitude = abs(value)
    for factor, prefix in _PREFIXES:
        if magnitude >= factor * 0.9995:
            return f"{value / factor:.{digits}g} {prefix}{unit}".strip()
    factor, prefix = _PREFIXES[-1]
    return f"{value / factor:.{digits}g} {prefix}{unit}".strip()


def fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= 1e9:
        return f"{value / 1e9:.2f} B"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if value >= 1e4:
        return f"{value / 1e3:.1f} k"
    return f"{int(value):,}"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def tidy(value: float) -> str:
    """A number as a person would type it: no float noise, no trailing zeros."""
    if not math.isfinite(value):
        return "0"
    text = f"{value:.10g}"
    if text in ("-0", "-0.0"):
        return "0"
    return text
