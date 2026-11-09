"""Board colours from the theme: per net, per differential pair, and the ports' ink."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from PySide6.QtGui import QColor

from EMShrimulator.app import theme

# Twelve well-spaced hues: dark tones on light themes, light ones on dark themes.
_NETS_LIGHT = ["#d9480f", "#1c7ed6", "#2b8a3e", "#ae3ec9", "#e67700", "#0c8599",
               "#c2255c", "#5c940d", "#4263eb", "#a61e4d", "#087f5b", "#862e9c"]
_NETS_DARK = ["#ff8a5c", "#74c0fc", "#8ce99a", "#e599f7", "#ffd43b", "#66d9e8",
              "#faa2c1", "#c0eb75", "#91a7ff", "#ff8787", "#63e6be", "#d0bfff"]
# Grounds and planes get a quiet slate.
_GROUND = {"light": "#a3acba", "dark": "#4a5a70"}
_NO_NET = {"light": "#c4cad4", "dark": "#3a4657"}
_GROUND_RE = re.compile(r"^(?:[ADPSC]?GND\w*|\w*_?GND|GROUND\w*|VSS\w*|0V|EARTH|CHASSIS)$", re.IGNORECASE)

# Differential pair colours, distinct from the nets'.
_PAIRS_LIGHT = ["#7048e8", "#0b7285", "#c92a2a", "#5f3dc4", "#2f9e44", "#d6336c"]
_PAIRS_DARK = ["#b197fc", "#3bc9db", "#ff8787", "#9775fa", "#69db7c", "#f783ac"]


def _light() -> bool:
    return theme.manager().is_light()


def is_ground(net: str) -> bool:
    return bool(_GROUND_RE.match(net.strip("/").rsplit("/", 1)[-1]))


def net_colors(nets: Iterable[str]) -> Dict[str, str]:
    """A colour for each net in order; grounds share the slate."""
    hues = _NETS_LIGHT if _light() else _NETS_DARK
    ground = _GROUND["light" if _light() else "dark"]
    out: Dict[str, str] = {}
    i = 0
    for net in nets:
        if is_ground(net):
            out[net] = ground
        else:
            out[net] = hues[i % len(hues)]
            i += 1
    return out


def no_net_color() -> str:
    """Copper that belongs to no net: a pad nobody wired, a logo."""
    return _NO_NET["light" if _light() else "dark"]


def pair_color(index: int) -> QColor:
    hues = _PAIRS_LIGHT if _light() else _PAIRS_DARK
    return QColor(hues[index % len(hues)])


def dimmed(color: QColor, backing: Optional[QColor] = None, amount: float = 0.8) -> QColor:
    """`color` faded toward the board's backing: copper not on the highlighted net."""
    backing = backing or QColor(theme.colors()["sunken"])
    return QColor.fromRgbF(*(a + (b - a) * amount for a, b in zip(color.getRgbF()[:3], backing.getRgbF()[:3])))


class PortInk:
    """The port ink from the theme: text colour, a contrasting halo, and the selection colour."""

    def __init__(self):
        c = theme.colors()
        light = _light()
        self.ink = QColor(c["text"])
        self.halo = QColor(c["raised"] if light else c["bg"])
        self.paper = QColor(c["raised"] if light else c["panel"])
        self.dim = QColor(c["text_dim"])
        self.select = QColor(c["selection"])
        self.warn = QColor(c["warn"])


def legend(nets: List[Dict]) -> List[Dict]:
    """The board's nets as the tree lists them: name, colour, the layers it is on."""
    colors = net_colors(net["name"] for net in nets)
    return [{"name": net["name"], "color": colors[net["name"]], "layers": net.get("layers", [])} for net in nets]
