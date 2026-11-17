"""Clicks in the 3D view reach the viewport even though the camera style grabs the mouse.

Needs OpenGL, so it skips on Qt's offscreen platform. Run with
`.venv/bin/python -m pytest -q tests/test_viewport_clicks.py`.
"""
import numpy as np
import pytest

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def viewport():
    app = QCoreApplication.instance()
    if app is None:
        QApplication.setOrganizationName("EMShrimulatorTests")  # snap settings must not land in the user's
        app = QApplication([])
    if not isinstance(app, QApplication) or QGuiApplication.platformName() == "offscreen":
        pytest.skip("needs an OpenGL window: run this file on its own, on the default Qt platform")
    from EMShrimulator.app.views.viewport3d import Viewport3D

    vp = Viewport3D()
    vp._window.SetOffScreenRendering(1)
    vp._window.SetSize(800, 600)
    vp._iren.SetSize(800, 600)
    vp.redraw()
    yield vp
    vp.close_plotter()


def _click(vp, x=400, y=300, button="Left"):
    vp._iren.SetEventInformation(x, y, 0, 0, chr(0), 0, None)
    getattr(vp._iren, f"{button}ButtonPressEvent")()
    getattr(vp._iren, f"{button}ButtonReleaseEvent")()


def test_a_click_is_heard(viewport):
    picked = []
    viewport.picked.connect(lambda kind, key: picked.append((kind, key)))
    _click(viewport)
    assert picked == [("", "")]  # empty space: the selection clears
    assert viewport._style.GetState() == 0  # and the style's orbit, started by the press, has ended


def test_a_tool_takes_the_click(viewport):
    from EMShrimulator.core.em import drawing

    finished = []
    viewport.pickFinished.connect(lambda tool, points: finished.append(points))
    plane = drawing.Plane.of_frame(np.zeros(3), np.eye(3), "xy")
    viewport.begin_pick(drawing.OriginTool(plane, "offset"))
    _click(viewport)
    assert len(finished) == 1 and not viewport.picking


def test_a_right_click_asks_for_the_menu(viewport):
    asked = []
    viewport.contextRequested.connect(lambda kind, key, where: asked.append(kind))
    _click(viewport, button="Right")
    for _ in range(5):
        QCoreApplication.processEvents()  # the menu opens after VTK finishes with the release
    assert asked == [""]
