"""The start page: recent projects on the left, the logo on the right. Project actions live in the File menu."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPixmap, QRadialGradient
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import theme
from EMShrimulator.app.notify import guarded, notify
from EMShrimulator.config import APP_ICON, APP_NAME, APP_VERSION, projects_dir
from EMShrimulator.core.services import projects

INFO = Qt.UserRole
LOGO_OPACITY = 0.5
LOGO_MARGIN = 6.0  # SVG units around the mark, so no stroke is clipped


def _when(stamp: float) -> str:
    """Today 21:33, Yesterday 18:02, 5 Oct, or 12 Sep 2025."""
    day = date.fromtimestamp(stamp)
    age = (date.today() - day).days
    if age in (0, 1):
        return f"{'Today' if age == 0 else 'Yesterday'} {time.strftime('%H:%M', time.localtime(stamp))}"
    return f"{day.day} {day:%b}" if day.year == date.today().year else f"{day.day} {day:%b %Y}"


class _RecentDelegate(QStyledItemDelegate):
    """Two lines per project: name and date; type, description and whether it has results."""

    PAD_X, PAD_Y, GAP = 12, 9, 3

    @staticmethod
    def _fonts(option) -> Tuple[QFont, QFont]:
        name = QFont(option.font)
        name.setPointSizeF(option.font.pointSizeF() * 1.2)
        name.setBold(True)
        return name, option.font

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt override
        name, small = self._fonts(option)
        return QSize(0, QFontMetrics(name).height() + self.GAP + QFontMetrics(small).height() + 2 * self.PAD_Y)

    def paint(self, painter: QPainter, option, index) -> None:
        info = index.data(INFO)
        c = theme.colors()
        name_font, small_font = self._fonts(option)
        name_fm, small_fm = QFontMetrics(name_font), QFontMetrics(small_font)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        box = QRectF(option.rect).adjusted(0.5, 1.5, -0.5, -1.5)
        if option.state & QStyle.State_Selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(c["accent_bg"]))
            painter.drawRoundedRect(box, 6, 6)
        elif option.state & QStyle.State_MouseOver:
            fill = QColor(c["raised"])
            fill.setAlpha(150)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(box, 6, 6)

        inner = option.rect.adjusted(self.PAD_X, self.PAD_Y, -self.PAD_X, -self.PAD_Y)
        top = QRect(inner.left(), inner.top(), inner.width(), name_fm.height())
        bottom = QRect(inner.left(), top.bottom() + 1 + self.GAP, inner.width(), small_fm.height())

        when = _when(info.get("modified", 0))
        painter.setFont(small_font)
        painter.setPen(QColor(c["text_faint"]))
        painter.drawText(top, Qt.AlignRight | Qt.AlignVCenter, when)
        room = top.width() - small_fm.horizontalAdvance(when) - 16
        painter.setFont(name_font)
        painter.setPen(QColor(c["text"]))
        painter.drawText(top, Qt.AlignLeft | Qt.AlignVCenter, name_fm.elidedText(info["name"], Qt.ElideRight, room))

        painter.setFont(small_font)
        kind = "Modeler" if info["kind"] == "modeler" else "Layout"
        painter.setPen(QColor(c["accent"]))
        painter.drawText(bottom, Qt.AlignLeft | Qt.AlignVCenter, kind)
        left = bottom.left() + small_fm.horizontalAdvance(kind) + 10
        right = bottom.right()
        if info.get("has_results"):
            painter.setPen(QColor(c["ok"]))
            painter.drawText(bottom, Qt.AlignRight | Qt.AlignVCenter, "results")
            right -= small_fm.horizontalAdvance("results") + 16
        description = " ".join((info.get("description") or "No description.").split())
        painter.setPen(QColor(c["text_dim"]))
        painter.drawText(QRect(left, bottom.top(), right - left, bottom.height()), Qt.AlignLeft | Qt.AlignVCenter,
                         small_fm.elidedText(description, Qt.ElideRight, right - left))
        painter.restore()


class ProjectGallery(QWidget):
    openRequested = Signal(str)
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._actions: List[Optional[QAction]] = []
        self._logo = QSvgRenderer(str(APP_ICON), self)
        self._mark = self._logo.boundsOnElement("mark").adjusted(-LOGO_MARGIN, -LOGO_MARGIN, LOGO_MARGIN, LOGO_MARGIN)
        self._logo_pixmap: Optional[QPixmap] = None
        self._logo_key: Optional[Tuple[int, int, float]] = None

        title = QLabel(APP_NAME)
        title.setProperty("role", "display")
        version = QLabel(f"Version {APP_VERSION} · EM simulation over openEMS")
        version.setProperty("role", "note")
        rule = QFrame()
        rule.setProperty("role", "rule")
        rule.setFixedSize(64, 3)
        self.lede = QLabel()
        self.lede.setProperty("role", "lede")
        self.lede.setWordWrap(True)
        self.hint = QLabel()
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)
        self.heading = QLabel("Recent projects")
        self.heading.setProperty("role", "heading")

        self.recent = QListWidget()
        self.recent.setProperty("role", "recent")
        self.recent.setFrameShape(QFrame.NoFrame)
        self.recent.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recent.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.recent.setUniformItemSizes(True)
        self.recent.setMouseTracking(True)
        self.recent.viewport().setAttribute(Qt.WA_Hover)
        self.recent.setItemDelegate(_RecentDelegate(self.recent))
        self.recent.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._context_menu)
        self.recent.itemActivated.connect(self._activated)
        self.recent.itemSelectionChanged.connect(self.selectionChanged)

        self.footer = QLabel()
        self.footer.setProperty("role", "note")
        self.footer.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.column = QWidget()
        self.column.setMinimumWidth(400)
        self.column.setMaximumWidth(600)
        col = QVBoxLayout(self.column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(title)
        col.addWidget(version)
        col.addSpacing(14)
        col.addWidget(rule)
        col.addSpacing(28)
        col.addWidget(self.lede)
        col.addSpacing(6)
        col.addWidget(self.hint)
        col.addSpacing(32)
        col.addWidget(self.heading)
        col.addSpacing(6)
        col.addWidget(self.recent, 1)
        col.addStretch(0)
        col.addSpacing(10)
        col.addWidget(self.footer)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(56, 44, 40, 22)
        layout.addWidget(self.column, 1)
        layout.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------- listing

    def refresh(self) -> None:
        rows = projects.list_projects()
        kept = self.selected()
        self.recent.blockSignals(True)
        self.recent.clear()
        for info in rows:
            item = QListWidgetItem()
            item.setData(INFO, info)
            item.setToolTip(f"{info['name']}\n{info.get('objects', 0)} objects · {info.get('ports', 0)} ports\n"
                            f"Modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(info.get('modified', 0)))}")
            self.recent.addItem(item)
            if kept and info["name"] == kept["name"]:
                item.setSelected(True)
                self.recent.setCurrentItem(item)
        self.recent.blockSignals(False)

        new = QKeySequence(QKeySequence.New).toString(QKeySequence.NativeText)
        self.lede.setText("Welcome back. Pick up where you left off, or start something new." if rows
                          else "Welcome. Let's simulate something.")
        self.hint.setText(
            f"<b>File › New project</b> ({new}) starts a parametric 3D Modeler design or a 3D Layout board from a "
            + ("template. Double-click a project below to open it." if rows else
               "template. Each one is a real, physically designed structure that runs as-is."))
        self.footer.setText(f"{len(rows)} project{'s' if len(rows) != 1 else ''} in {projects_dir()}")
        self.heading.setVisible(bool(rows))
        self.recent.setVisible(bool(rows))
        self.selectionChanged.emit()

    def selected(self) -> Optional[Dict]:
        items = self.recent.selectedItems()
        return items[0].data(INFO) if items else None

    def use_actions(self, actions: List[Optional[QAction]]) -> None:
        """Use the File menu's project actions in the right-click menu (None is a separator)."""
        self._actions = actions
        self.recent.addActions([a for a in actions if a is not None])

    def _context_menu(self, pos) -> None:
        item = self.recent.itemAt(pos)
        if item is None or not self._actions:
            return
        self.recent.setCurrentItem(item)
        menu = QMenu(self)
        for action in self._actions:
            if action is None:
                menu.addSeparator()
            else:
                menu.addAction(action)
        menu.exec(self.recent.viewport().mapToGlobal(pos))

    # ---------------------------------------------------------------- logo

    def _logo_rect(self) -> QRectF:
        """Centred right of the text column, as large as fits."""
        room = max(self.width() - self.column.geometry().right(), self.width() * 0.4)
        scale = min(room * 0.8 / self._mark.width(), self.height() * 0.66 / self._mark.height())
        # In 16 px steps, so a drag-resize doesn't re-render every frame.
        w = max(16.0, self._mark.width() * scale // 16 * 16)
        h = w * self._mark.height() / self._mark.width()
        return QRectF(self.width() - room / 2 - w / 2, (self.height() - h) / 2, w, h)

    def _logo_at(self, size: QSize) -> QPixmap:
        # Rendered once per size.
        ratio = self.devicePixelRatioF()
        key = (size.width(), size.height(), ratio)
        if key != self._logo_key:
            pixmap = QPixmap(round(size.width() * ratio), round(size.height() * ratio))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            scale = size.width() / self._mark.width()
            margin = LOGO_MARGIN * scale
            self._logo.render(painter, "mark",
                              QRectF(margin, margin, size.width() - 2 * margin, size.height() - 2 * margin))
            painter.end()
            self._logo_pixmap, self._logo_key = pixmap, key
        return self._logo_pixmap

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        area = self._logo_rect()
        target = area.toAlignedRect()
        if target.width() < 40:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        glow = QRadialGradient(area.center(), max(area.width(), area.height()) * 0.75)
        tint = QColor(theme.colors()["accent"])
        for stop, alpha in ((0.0, 44), (0.55, 16), (1.0, 0)):
            tint.setAlpha(alpha)
            glow.setColorAt(stop, tint)
        painter.fillRect(event.rect(), glow)
        painter.setOpacity(LOGO_OPACITY)
        painter.drawPixmap(target.topLeft(), self._logo_at(target.size()))

    # ------------------------------------------------------------- actions

    def _activated(self, item: QListWidgetItem) -> None:
        self.openRequested.emit(item.data(INFO)["name"])

    def open_selected(self) -> None:
        info = self.selected()
        if info:
            self.openRequested.emit(info["name"])

    def rename_selected(self) -> None:
        info = self.selected()
        if not info:
            return
        new, ok = QInputDialog.getText(self, "Rename project", "New name:", text=info["name"])
        if ok and new.strip() and new.strip() != info["name"]:
            if guarded(projects.rename_project, info["name"], new.strip()) is not None:
                notify("Renamed.", "ok")
                self.refresh()

    def duplicate_selected(self) -> None:
        info = self.selected()
        if info and guarded(projects.duplicate_project, info["name"], projects.suggest_copy_name(info["name"])):
            notify(f"Duplicated \"{info['name']}\".", "ok")
            self.refresh()

    def delete_selected(self) -> None:
        info = self.selected()
        if not info:
            return
        answer = QMessageBox.warning(
            self, "Delete project",
            f"Delete \"{info['name']}\" and everything in it — geometry, job logs and results? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Yes:
            guarded(projects.delete_project, info["name"])
            notify(f"Deleted \"{info['name']}\".", "ok")
            self.refresh()

    def export_selected(self) -> None:
        info = self.selected()
        if not info:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Export project", str(Path.home() / f"{info['name']}.gems.zip"),
                                                "EMShrimulator project (*.gems.zip *.zip)")
        if target and guarded(projects.export_to_file, info["name"], Path(target)):
            notify(f"Exported to {target}.", "ok")

    def import_project(self) -> None:
        source, _ = QFileDialog.getOpenFileName(self, "Import project", str(Path.home()),
                                                "EMShrimulator project (*.zip)")
        if source:
            info = guarded(projects.import_from_file, Path(source))
            if info:
                notify(f"Imported \"{info['name']}\".", "ok")
                self.refresh()
