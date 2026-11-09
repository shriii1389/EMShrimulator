"""Create a project from a template or a bundled example board."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from EMShrimulator.app.widgets.form import Segmented, hint
from EMShrimulator.core.services import projects

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()\-]{0,63}$")


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.resize(820, 560)
        self.templates: Dict[str, List[Dict]] = projects.list_templates()
        self.existing = {p["name"] for p in projects.list_projects()}
        self._name_touched = False

        self.kind = Segmented([("modeler", "3D Modeler"), ("layout", "3D Layout")])
        self.kind.set_value("modeler")
        self.kind.committed.connect(lambda _k: self._fill_templates())
        self.name = QLineEdit()
        self.name.setPlaceholderText("Project name")
        self.name.textEdited.connect(self._name_edited)
        self.name_error = QLabel()
        self.name_error.setProperty("role", "error")
        self.explanation = hint()
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setSpacing(2)
        self.list.currentItemChanged.connect(lambda *_: self._template_changed())
        self.list.itemDoubleClicked.connect(lambda _item: self.accept() if self._valid() else None)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.kind)
        layout.addWidget(self.name)
        layout.addWidget(self.name_error)
        layout.addWidget(self.explanation)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.buttons)
        self._fill_templates()

    def chosen_kind(self) -> str:
        return str(self.kind.value())

    def chosen_template(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def chosen_name(self) -> str:
        return self.name.text().strip()

    def _fill_templates(self) -> None:
        kind = self.chosen_kind()
        self.explanation.setText(
            "Every template is a real structure with dimensions worked out from the standard design equations — "
            "create one and press Run without editing anything." if kind == "modeler" else
            "Each example is one of gerber2ems's own bundled KiCad boards, imported with its stackup and its ports "
            "already placed. Choose Blank board to import your own Gerbers.")
        self.list.clear()
        entries = list(self.templates.get(kind, []))
        if kind == "layout":
            entries.insert(0, {"id": None, "title": "Blank board", "summary": "Start empty and import a Gerber "
                               "zip or an ODB++ model.", "tags": []})
        for entry in entries:
            tags = "   ·   ".join(entry.get("tags") or [])
            item = QListWidgetItem(f"{entry['title']}\n{entry['summary']}" + (f"\n{tags}" if tags else ""))
            item.setData(Qt.UserRole, entry["id"])
            item.setData(Qt.UserRole + 1, entry["title"])
            self.list.addItem(item)
        self._name_touched = False
        self.list.setCurrentRow(0)

    def _template_changed(self) -> None:
        if not self._name_touched:
            item = self.list.currentItem()
            base = re.sub(r"[^A-Za-z0-9 _.-]", "", item.data(Qt.UserRole + 1) if item else "New design").strip()
            base = base or "New design"
            candidate, n = base, 2
            while candidate in self.existing:
                candidate = f"{base} {n}"
                n += 1
            self.name.setText(candidate)
        self._valid()

    def _name_edited(self, _text: str) -> None:
        self._name_touched = True
        self._valid()

    def _valid(self) -> bool:
        name = self.chosen_name()
        if not name:
            error = "Give the project a name."
        elif name in self.existing:
            error = "A project with that name already exists."
        elif not NAME_RE.match(name):
            error = "Letters, digits, spaces and . - _ ( ) only, starting with a letter or digit."
        else:
            error = ""
        self.name_error.setText(error)
        self.name_error.setVisible(bool(error))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not error)
        return not error
