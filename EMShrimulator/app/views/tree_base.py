"""The project tree base: items keyed by tuples, rebuilt only when their content changes.

Subclasses implement `signature`, `build` and `selection_key`.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from EMShrimulator.app import theme
from EMShrimulator.app.widgets.icons import icon, swatch

KEY_ROLE = Qt.UserRole + 7
TEXT_ROLE = Qt.UserRole + 8
Key = Tuple


class ProjectTreeBase(QWidget):
    navigate = Signal(object)              # key tuple
    menuRequested = Signal(object, object)  # key tuple, global QPoint

    # Item kinds that stand for a selection.
    selection_kinds: Tuple[str, ...] = ()
    # Kinds of item left folded the first time the tree is built.
    folded_kinds: Tuple[str, ...] = ()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False
        self._signature = None
        self._items: Dict[Key, QTreeWidgetItem] = {}
        self._badges: Dict[Key, str] = {}
        self._first_build = True
        self._pending: Optional[Key] = None

        self.tree = QTreeWidget()
        self.tree.setObjectName("project-tree")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.currentItemChanged.connect(lambda item, _prev: self._activate(item))
        self.tree.itemClicked.connect(lambda item, _col: self._activate(item))
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tree)

        session.projectChanged.connect(self._schedule_refresh)
        session.selectionChanged.connect(self._selection_changed)
        theme.manager().changed.connect(self._theme_changed)
        self.refresh()

    # ------------------------------------------------------- for subclasses

    def signature(self) -> tuple:
        """Everything the tree shows: it is rebuilt when this changes."""
        raise NotImplementedError

    def build(self) -> None:
        """Add the items, through `_item`."""
        raise NotImplementedError

    def selection_key(self, selection) -> Key:
        """The item a selection stands for."""
        return selection.kind, selection.key

    def item_checked(self, key: Key, checked: bool) -> None:
        """The user ticked or unticked a checkable item."""

    # -------------------------------------------------------------- public

    def set_badge(self, key: Key, text: str) -> None:
        """A live suffix on one item's label - a count or a run state - kept across rebuilds."""
        if self._badges.get(key) == text:
            return
        self._badges[key] = text
        item = self._items.get(key)
        if item is not None:
            self._syncing = True
            try:
                item.setText(0, self._label_with_badge(key, item.data(0, TEXT_ROLE)))
            finally:
                self._syncing = False

    def set_current(self, key: Optional[Key]) -> None:
        """Highlight an item without navigating; one not built yet is highlighted once it exists."""
        item = self._items.get(key) if key is not None else None
        self._pending = key if key is not None and item is None else None
        self._syncing = True
        try:
            if item is None:
                self.tree.setCurrentItem(None)
                self.tree.clearSelection()
            else:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
        finally:
            self._syncing = False

    def current_key(self) -> Optional[Key]:
        item = self.tree.currentItem()
        return item.data(0, KEY_ROLE) if item is not None else None

    def keys(self) -> Iterable[Key]:
        return self._items.keys()

    # ------------------------------------------------------------- refresh

    def _schedule_refresh(self) -> None:
        QTimer.singleShot(0, self.refresh)

    def _theme_changed(self, _name: str) -> None:
        self._signature = None
        self.refresh()

    def refresh(self) -> None:
        signature = self.signature()
        if signature == self._signature:
            return
        self._signature = signature
        expanded = {key for key, item in self._items.items() if item.isExpanded()}
        current = self.current_key()
        scroll = self.tree.verticalScrollBar().value()

        self._syncing = True
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            self._items = {}
            self.build()
            for key, item in self._items.items():
                if self._first_build:
                    item.setExpanded(key[0] not in self.folded_kinds)
                else:
                    item.setExpanded(key in expanded)
            self._first_build = False
            if current in self._items:
                self.tree.setCurrentItem(self._items[current])
        finally:
            self.tree.setUpdatesEnabled(True)
            self._syncing = False
        self.tree.verticalScrollBar().setValue(scroll)
        if self._pending in self._items:
            self.set_current(self._pending)

    # --------------------------------------------------------------- build

    def _item(self, parent, key: Key, text: str, icon_name: str = "", *, tooltip: str = "",
              color: Optional[str] = None, bold: bool = False, swatch_color: Optional[str] = None,
              selectable: bool = True, checked: Optional[bool] = None) -> QTreeWidgetItem:
        c = theme.colors()
        item = QTreeWidgetItem(parent)
        item.setData(0, KEY_ROLE, key)
        item.setData(0, TEXT_ROLE, text)
        item.setText(0, self._label_with_badge(key, text))
        if swatch_color:
            item.setIcon(0, swatch(swatch_color))
        elif icon_name:
            item.setIcon(0, icon(icon_name, c["text_dim"], c["accent"]))
        if tooltip:
            item.setToolTip(0, tooltip)
        if color:
            item.setForeground(0, QColor(color))
        if bold:
            font = item.font(0)
            font.setWeight(QFont.DemiBold)
            item.setFont(0, font)
        if not selectable:
            item.setFlags(Qt.ItemIsEnabled)
        if checked is not None:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self._items[key] = item
        return item

    def _label_with_badge(self, key: Key, text: str) -> str:
        badge = self._badges.get(key)
        return f"{text}  ·  {badge}" if badge else text

    # -------------------------------------------------------------- events

    def _activate(self, item: Optional[QTreeWidgetItem]) -> None:
        if self._syncing or item is None:
            return
        key = item.data(0, KEY_ROLE)
        if key is not None and item.flags() & Qt.ItemIsSelectable:
            self.navigate.emit(key)

    def _item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._syncing:
            return
        key = item.data(0, KEY_ROLE)
        if key and item.flags() & Qt.ItemIsUserCheckable:
            self.item_checked(key, item.checkState(0) == Qt.Checked)

    def _selection_changed(self, selection) -> None:
        if selection is None:
            current = self.current_key()
            if current and current[0] in self.selection_kinds:
                self.set_current(None)
            return
        key = self.selection_key(selection)
        if key in self._items:
            self.set_current(key)

    def _context_menu(self, pos: QPoint) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        key = item.data(0, KEY_ROLE)
        if key is not None:
            self.menuRequested.emit(key, self.tree.viewport().mapToGlobal(pos))
