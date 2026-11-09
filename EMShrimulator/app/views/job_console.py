"""Run log, progress and run history. A QTimer polls the job log, so nothing blocks."""
from __future__ import annotations

import re
import time
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import theme
from EMShrimulator.app.widgets.form import fmt_duration
from EMShrimulator.core import jobs

POLL_MS = 700
FINISHED = ("done", "failed", "cancelled")
MAX_LOG_LINES = 50_000

_ERROR = re.compile(r"\bERROR\b|\bERRO\b|Traceback|Exception|\[ERROR\]")


class JobMonitor(QObject):
    """Follows one job: log text as it arrives, status as it changes."""

    logAppended = Signal(str)
    statusChanged = Signal(dict)
    finished = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job_id: Optional[str] = None
        self._offset = 0
        self._last_status = ""
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.poll)

    def follow(self, job_id: str, from_start: bool = True) -> None:
        self.job_id = job_id
        self._offset = 0 if from_start else self._offset
        self._last_status = ""
        self.poll()
        job = jobs.get_job(job_id)
        if job is not None and job.status not in FINISHED:
            self._timer.start()

    def stop_following(self) -> None:
        self._timer.stop()
        self.job_id = None

    @property
    def running(self) -> bool:
        job = jobs.get_job(self.job_id) if self.job_id else None
        return job is not None and job.status not in FINISHED

    def poll(self) -> None:
        if not self.job_id:
            return
        job = jobs.get_job(self.job_id)
        if job is None:
            self._timer.stop()
            return
        text, self._offset = jobs.read_log(self.job_id, self._offset)
        if text:
            self.logAppended.emit(text)
        status = job.to_dict()
        self.statusChanged.emit(status)
        if job.status in FINISHED and self._last_status not in FINISHED:
            self._timer.stop()
            # Final read: the process may exit between the read and the status check.
            tail, self._offset = jobs.read_log(self.job_id, self._offset)
            if tail:
                self.logAppended.emit(tail)
            self.finished.emit(job.to_dict())
        self._last_status = job.status


class LogView(QPlainTextEdit):
    """Colourised engine output that follows the tail unless scrolled up."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setProperty("role", "log")
        self.setMaximumBlockCount(MAX_LOG_LINES)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setPlaceholderText("No output yet. Press Run to start a simulation - the engine's live output appears here.")
        self._partial = ""

    def clear_log(self) -> None:
        self._partial = ""
        self.clear()

    def append_text(self, text: str) -> None:
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        text = self._partial + text
        lines = text.split("\n")
        self._partial = lines.pop()  # an unterminated last line waits for the rest
        colors = theme.colors()
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End)
        for line in lines:
            if line.startswith("@@GEMS "):
                continue  # structured progress markers drive the progress bar instead
            fmt = QTextCharFormat()
            upper = line.upper()
            if _ERROR.search(line):
                fmt.setForeground(QColor(colors["danger"]))
            elif "WARNING" in upper or "[WARN" in upper:
                fmt.setForeground(QColor(colors["warn"]))
            elif line.startswith("===") or line.startswith("RUN_COMPLETE") or "COMPLETE" in upper:
                fmt.setForeground(QColor(colors["ok"]))
            elif line.startswith("$ ") or "Timestep" in line:
                fmt.setForeground(QColor(colors["accent"]))
            else:
                fmt.setForeground(QColor(colors["text_dim"]))
            cursor.insertText(line + "\n", fmt)
        if at_bottom:
            bar.setValue(bar.maximum())


class JobStatusBar(QWidget):
    """Stage, progress, elapsed time and a Stop button for the followed job."""

    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        self.dot = QLabel("●")
        self.stage = QLabel("Idle")
        self.stage.setMinimumWidth(160)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(140)
        self.elapsed = QLabel("")
        self.elapsed.setMinimumWidth(56)
        self.stop = QPushButton("■ Stop")
        self.stop.setProperty("role", "danger")
        self.stop.clicked.connect(self.stopRequested)
        layout.addWidget(self.dot)
        layout.addWidget(self.stage, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.elapsed)
        layout.addWidget(self.stop)
        self.show_status(None)

    def show_status(self, status: Optional[dict]) -> None:
        colors = theme.colors()
        self.progress.setVisible(bool(status))
        if not status:
            self.dot.setStyleSheet(f"color: {colors['text_faint']}")
            self.stage.setText("No run yet")
            self.progress.setValue(0)
            self.elapsed.setText("")
            self.stop.setVisible(False)
            return
        running = status["status"] not in FINISHED
        color = colors["accent"] if running else colors["ok"] if status["status"] == "done" else colors["danger"]
        self.dot.setStyleSheet(f"color: {color}")
        self.stage.setText(f"{status.get('label') or ''} — {status.get('stage') or status['status']}")
        progress = status.get("progress")
        if running and progress is None:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int((progress or 0) * 1000))
        self.elapsed.setText(fmt_duration(status.get("elapsed_s")))
        self.stop.setVisible(running)


class JobConsole(QWidget):
    """Status strip over the log of the job being followed."""

    finished = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.monitor = JobMonitor(self)
        self.status = JobStatusBar()
        self.log = LogView()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.status)
        layout.addWidget(self.log, 1)

        self.monitor.logAppended.connect(self.log.append_text)
        self.monitor.statusChanged.connect(self.status.show_status)
        self.monitor.finished.connect(self.finished)
        self.status.stopRequested.connect(self.cancel)

    def follow(self, job_id: str) -> None:
        self.log.clear_log()
        self.monitor.follow(job_id)

    def cancel(self) -> None:
        if self.monitor.job_id:
            jobs.cancel_job(self.monitor.job_id)
            self.monitor.poll()

    @property
    def running(self) -> bool:
        return self.monitor.running


class RunsView(QWidget):
    """Every run of this project in this session, newest first."""

    openLog = Signal(str)

    COLUMNS = ["Status", "Started", "Elapsed", "Stage", "What ran"]

    def __init__(self, project_name: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open)
        self.empty = QLabel("No runs in this session yet.")
        self.empty.setProperty("role", "hint")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.empty)
        layout.addWidget(self.table, 1)
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def refresh(self) -> None:
        if not self.isVisible() and self.table.rowCount():
            return
        items = jobs.list_jobs(self.project_name)
        self.empty.setVisible(not items)
        self.table.setVisible(bool(items))
        self.table.setRowCount(len(items))
        colors = theme.colors()
        for row, job in enumerate(items):
            started = time.strftime("%H:%M:%S", time.localtime(job["started_at"])) if job["started_at"] else "—"
            values = [job["status"], started, fmt_duration(job["elapsed_s"]), job["stage"] or "—",
                      job["label"] or job["kind"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, job["id"])
                if column == 0:
                    tint = {"done": "ok", "failed": "danger", "running": "accent", "cancelled": "warn"}.get(job["status"])
                    if tint:
                        item.setForeground(QColor(colors[tint]))
                self.table.setItem(row, column, item)

    def _open(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.openLog.emit(item.data(Qt.UserRole))
