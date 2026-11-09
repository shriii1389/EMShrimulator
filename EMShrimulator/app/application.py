"""Application start-up. Heavy imports happen after the splash is shown."""
from __future__ import annotations

import sys
import traceback
from typing import Dict, List, Optional

from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from EMShrimulator.config import APP_ICON, APP_NAME, APP_VERSION, ORGANISATION, set_projects_dir


def _splash(c: Dict[str, str]) -> QSplashScreen:
    pixmap = QPixmap(560, 220)
    pixmap.fill(QColor(c["bg"]))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(str(APP_ICON)).render(painter, QRectF(24, 46, 112, 112))
    painter.setPen(QColor(c["accent"]))
    painter.setFont(QFont("Segoe UI", 22, QFont.Bold))
    painter.drawText(156, 92, APP_NAME)
    painter.setPen(QColor(c["text_dim"]))
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(158, 124, f"Version {APP_VERSION} · EM simulation over openEMS")
    painter.fillRect(158, 138, 64, 3, QColor(c["primary"]))
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.showMessage("Loading the 3D viewport and solver interfaces…", Qt.AlignBottom | Qt.AlignLeft,
                       QColor(c["text_faint"]))
    return splash


def _install_excepthook() -> None:
    """An unexpected exception in a slot is reported, not silently swallowed."""
    previous = sys.excepthook

    def hook(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        print(text, file=sys.stderr)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox(QMessageBox.Critical, "Unexpected error",
                              f"{value}\n\nThe application is still running; the details below are worth "
                              "including in a bug report.")
            box.setDetailedText(text)
            box.exec()
        else:
            previous(kind, value, tb)

    sys.excepthook = hook


def run(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    QApplication.setOrganizationName(ORGANISATION)
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)
    if sys.platform == "win32":
        # Own taskbar identity, so the taskbar shows the app icon.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{ORGANISATION}.Studio")
    app = QApplication(argv)
    app.setWindowIcon(QIcon(str(APP_ICON)))

    settings = QSettings()
    folder = settings.value("projects/folder")
    if folder:
        try:
            set_projects_dir(str(folder))
        except OSError:
            settings.remove("projects/folder")

    from EMShrimulator.app import theme

    theme.manager().apply(str(settings.value("appearance/theme") or theme.DEFAULT_THEME))

    splash = _splash(theme.colors())
    splash.show()
    app.processEvents()

    _install_excepthook()
    from EMShrimulator.app.mainwindow import MainWindow

    window = MainWindow()
    window.show()
    splash.finish(window)

    from EMShrimulator.app import preload

    preload.start()

    if "--open" in argv:
        index = argv.index("--open")
        if index + 1 < len(argv):
            window.open_project(argv[index + 1])
    try:
        return app.exec()
    finally:
        from EMShrimulator.core import procpool

        procpool.shutdown()
