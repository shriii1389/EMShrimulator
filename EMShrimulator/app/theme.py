"""Theme token tables for the style sheet, charts and VTK."""
from __future__ import annotations

from string import Template
from typing import Dict, List

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

from EMShrimulator.config import RESOURCES_DIR

THEMES: Dict[str, Dict[str, str]] = {
    # Poseidon navy surfaces, Norse Blue accent, Pureed Pumpkin for primary actions and formulas.
    "poseidon": {
        "label": "Poseidon",
        "bg": "#0c2337", "panel": "#0f2a42", "raised": "#123955", "sunken": "#081a2a",
        "viewport": "#091c2d", "line": "#234d6c", "line_strong": "#4c7b9c",
        "text": "#eaf2f8", "text_dim": "#b6c9d8", "text_faint": "#8ba5ba",
        "accent": "#4ca5c7", "accent_bg": "#1c5b7c", "primary": "#c34121",
        "ok": "#5cc28f", "warn": "#f0b44c",
        "danger": "#ef5f5f", "danger_bg": "#3d1a20", "formula": "#f2a072",
        "grid_major": "#3f6787", "grid_minor": "#1e4260", "mesh_line": "#6cc0de",
        "vp_sky": "#1a4566", "vp_floor": "#06131f", "vp_ground": "#0b2033",
        "vp_grid": "#2b5878", "vp_grid_major": "#4d82a8",
        "axis_x": "#ef6461", "axis_y": "#5fcf8a", "axis_z": "#5aaee0",
        "selection": "#ff5cf0", "snap": "#ffd43b", "snap_halo": "#04101c",
        "is_light": "0",
    },
    "graphite": {
        "label": "Graphite",
        "bg": "#16181e", "panel": "#1c1f26", "raised": "#272c35", "sunken": "#101216",
        "viewport": "#12141a", "line": "#3a414d", "line_strong": "#6b7488",
        "text": "#e4e8ef", "text_dim": "#b9bec8", "text_faint": "#8f97a7",
        "accent": "#4a9eff", "accent_bg": "#16427a", "primary": "#4a9eff",
        "ok": "#4caf7d", "warn": "#e0a53a",
        "danger": "#e05a5a", "danger_bg": "#3b1618", "formula": "#8fc3ff",
        "grid_major": "#4a5567", "grid_minor": "#2f353f", "mesh_line": "#5aa8f0",
        "vp_sky": "#272d38", "vp_floor": "#0d0f14", "vp_ground": "#1a1e26",
        "vp_grid": "#353c48", "vp_grid_major": "#56617a",
        "axis_x": "#e0555f", "axis_y": "#5fc27e", "axis_z": "#4f92e8",
        "selection": "#ff5cf0", "snap": "#ffd43b", "snap_halo": "#07080b",
        "is_light": "0",
    },
    "daylight": {
        "label": "Daylight",
        "bg": "#eef1f5", "panel": "#f8f9fb", "raised": "#ffffff", "sunken": "#e6e9ef",
        "viewport": "#dfe4ec", "line": "#c3cad6", "line_strong": "#768298",
        "text": "#161b23", "text_dim": "#454d5c", "text_faint": "#5b6579",
        "accent": "#1261b3", "accent_bg": "#dbe9fa", "primary": "#1261b3",
        "ok": "#1d7147", "warn": "#8a5a0b",
        "danger": "#bf2f2f", "danger_bg": "#fbe3e3", "formula": "#0d4f96",
        "grid_major": "#9aa4b4", "grid_minor": "#c3cad6", "mesh_line": "#2f6fb0",
        "vp_sky": "#f2f5fa", "vp_floor": "#c3ccda", "vp_ground": "#e8ecf3",
        "vp_grid": "#cbd2dd", "vp_grid_major": "#a3aec0",
        "axis_x": "#c0392b", "axis_y": "#2e8b57", "axis_z": "#2565b8",
        "selection": "#c026d3", "snap": "#c2410c", "snap_halo": "#ffffff",
        "is_light": "1",
    },
    "midnight": {
        "label": "Midnight",
        "bg": "#05080e", "panel": "#0b0f17", "raised": "#16202e", "sunken": "#02040a",
        "viewport": "#04060b", "line": "#2c3a4c", "line_strong": "#5c7490",
        "text": "#f2f6fc", "text_dim": "#aebbcc", "text_faint": "#8698ae",
        "accent": "#5cb0ff", "accent_bg": "#0d3f7a", "primary": "#5cb0ff",
        "ok": "#4bd48a", "warn": "#ffb23d",
        "danger": "#ff6b6b", "danger_bg": "#3a0d10", "formula": "#9ccfff",
        "grid_major": "#445a75", "grid_minor": "#202d3b", "mesh_line": "#6fbcff",
        "vp_sky": "#101a28", "vp_floor": "#010307", "vp_ground": "#0a111b",
        "vp_grid": "#1c2938", "vp_grid_major": "#33475f",
        "axis_x": "#ff5e6b", "axis_y": "#4ade8a", "axis_z": "#58a6ff",
        "selection": "#ff5cf0", "snap": "#ffd43b", "snap_halo": "#000000",
        "is_light": "0",
    },
}

# Categorical ink for chart traces: distinguishable in both light and dark.
TRACE_COLORS: List[str] = [
    "#4a9eff", "#f2913d", "#4cc38a", "#c46bd9", "#e5c04b",
    "#3fb8c4", "#e2647a", "#8a9ef0", "#9bbf4a", "#d97b3c",
]

DEFAULT_THEME = "poseidon"


def trace_color(index: int) -> str:
    return TRACE_COLORS[index % len(TRACE_COLORS)]


class ThemeManager(QObject):
    """Owns the active theme and tells every custom-painted view when it changes."""

    changed = Signal(str)

    def __init__(self, name: str = DEFAULT_THEME):
        super().__init__()
        self._name = name if name in THEMES else DEFAULT_THEME
        self._styled = False

    @property
    def name(self) -> str:
        return self._name

    def colors(self) -> Dict[str, str]:
        return THEMES[self._name]

    def is_light(self) -> bool:
        return THEMES[self._name]["is_light"] == "1"

    def apply(self, name: str) -> None:
        if name not in THEMES:
            name = DEFAULT_THEME
        self._name = name
        app = QApplication.instance()
        if app is None:
            return
        tokens = THEMES[name]
        # The style is set once; switching themes only swaps the palette and style sheet.
        if not self._styled:
            app.setStyle(QStyleFactory.create("Fusion"))
            self._styled = True
        app.setPalette(_palette(tokens))
        app.setStyleSheet(_style_template().safe_substitute(tokens))
        self.changed.emit(name)


_template: Template | None = None


def _style_template() -> Template:
    global _template
    if _template is None:
        _template = Template((RESOURCES_DIR / "themes" / "base.qss").read_text(encoding="utf-8"))
    return _template


def _palette(t: Dict[str, str]) -> QPalette:
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(t["bg"]))
    p.setColor(QPalette.WindowText, c(t["text"]))
    p.setColor(QPalette.Base, c(t["sunken"]))
    p.setColor(QPalette.AlternateBase, c(t["panel"]))
    p.setColor(QPalette.ToolTipBase, c(t["raised"]))
    p.setColor(QPalette.ToolTipText, c(t["text"]))
    p.setColor(QPalette.PlaceholderText, c(t["text_faint"]))
    p.setColor(QPalette.Text, c(t["text"]))
    p.setColor(QPalette.Button, c(t["raised"]))
    p.setColor(QPalette.ButtonText, c(t["text"]))
    p.setColor(QPalette.BrightText, c(t["danger"]))
    p.setColor(QPalette.Highlight, c(t["accent"]))
    p.setColor(QPalette.HighlightedText, c("#ffffff"))
    p.setColor(QPalette.Link, c(t["accent"]))
    p.setColor(QPalette.Mid, c(t["line"]))
    p.setColor(QPalette.Dark, c(t["sunken"]))
    p.setColor(QPalette.Light, c(t["line_strong"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(t["text_faint"]))
    return p


_manager: ThemeManager | None = None


def manager() -> ThemeManager:
    """The application's single theme manager."""
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager


def colors() -> Dict[str, str]:
    return manager().colors()
