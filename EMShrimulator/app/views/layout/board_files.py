"""The board files page: the imported files and the import itself."""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QPushButton, QScrollArea

from EMShrimulator.app.widgets.form import card, centred_column, hint, row

IMPORT_HINT = ("Import the whole Gerber export as a .zip — it is unpacked, each file is identified from its own layer "
               "attribute or its name, and everything is renamed to what the solver expects. KiCad, Altium and "
               "fab-house exports all work.")


class BoardFiles(QScrollArea):
    deleteFile = Signal(str)
    clearFiles = Signal()
    showReport = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setObjectName("setup-page")

        self.files = QListWidget()
        self.files.setMinimumHeight(220)
        self.files.setToolTip("Double-click a file to remove it")
        self.files.itemDoubleClicked.connect(lambda item: self.deleteFile.emit(item.data(Qt.UserRole))
                                             if item.data(Qt.UserRole) else None)
        self.empty = hint(IMPORT_HINT)
        self.source = hint(role="note")
        self.report_button = QPushButton("Import report")
        self.report_button.setToolTip("What the last import found in each file")
        self.report_button.clicked.connect(self.showReport)
        self.report_button.setVisible(False)
        self.clear_button = QPushButton("Clear all…")
        self.clear_button.setProperty("role", "danger")
        self.clear_button.setToolTip("Remove every board file. Ports and settings are kept.")
        self.clear_button.clicked.connect(self.clearFiles)

        frame, layout = card("Imported files", "The gerbers and the drill file the board is drawn and simulated "
                                               "from. Double-click one to remove it.")
        layout.addWidget(self.empty)
        layout.addWidget(self.files)
        layout.addWidget(self.source)
        layout.addWidget(row(self.report_button, self.clear_button))
        self.setWidget(centred_column([frame]))

    def show_files(self, project, drill_count: int) -> None:
        self.files.clear()
        names: List[str] = list(project.gerber_files)
        for name in names:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.files.addItem(item)
        if project.drill_file:
            item = QListWidgetItem(f"{project.drill_file}   ·  drill · {drill_count} holes")
            item.setData(Qt.UserRole, project.drill_file)
            self.files.addItem(item)
        has_files = bool(names or project.drill_file)
        self.files.setVisible(has_files)
        self.empty.setVisible(not has_files)
        self.clear_button.setEnabled(has_files)
        self.source.setText("Converted from an ODB++ product model." if project.source_format == "odb++" else "")
        self.source.setVisible(project.source_format == "odb++")
