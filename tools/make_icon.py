"""Rebuild packaging/EMShrimulator.ico from the shrimp SVG: python tools/make_icon.py"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

from EMShrimulator.config import APP_ICON  # noqa: E402

SIZES = [16, 24, 32, 48, 64, 128, 256]
OUT = ROOT / "packaging" / "EMShrimulator.ico"


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    buffer = QBuffer()
    buffer.open(QBuffer.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def main() -> int:
    app = QGuiApplication(sys.argv)  # noqa: F841 - QPainter needs a Qt application
    renderer = QSvgRenderer(str(APP_ICON))
    if not renderer.isValid():
        print(f"Could not read {APP_ICON}", file=sys.stderr)
        return 1
    images = [render(renderer, size) for size in SIZES]
    images[-1].save(OUT, format="ICO", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    print(f"Wrote {OUT.relative_to(ROOT)} ({', '.join(map(str, SIZES))} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
