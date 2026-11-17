"""2D board canvas: vector copper matches the rasters, clear polarity erases, and zooming in stays sharp."""
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QTransform  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from EMShrimulator.app.views.layout.board_canvas import BoardCanvas, decode_shapes, paint_shapes  # noqa: E402
from EMShrimulator.core.layout import board, gerber_shapes, templates  # noqa: E402

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


WORK = Path(tempfile.mkdtemp(prefix="gems-shapes-"))
QApplication.setOrganizationName("EMShrimulator-tests")  # keep QSettings writes out of the user's
app = QApplication.instance() or QApplication([])


def example(example_id):
    d = WORK / example_id
    if not d.exists():
        project = templates.build(example_id, example_id, d)
        (d / "project.json").write_text(project.model_dump_json(), encoding="utf-8")
    from EMShrimulator.core.layout.model import LayoutProject
    project = LayoutProject.model_validate_json((d / "project.json").read_text(encoding="utf-8"))
    return board.board_info(d, [layer.model_dump() for layer in project.stackup]), d


def vector_alpha(shapes, width_px, height_px, ppm_x, ppm_y, height_mm):
    """A layer's shapes painted on the raster's own pixel grid, as uint8 alpha."""
    image = QImage(width_px, height_px, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setWorldTransform(QTransform(ppm_x, 0, 0, -ppm_y, 0, height_mm * ppm_y))
    paint_shapes(painter, decode_shapes(shapes), QColor("white"), QRectF(-1e6, -1e6, 2e6, 2e6))
    painter.end()
    argb = np.frombuffer(image.constBits(), np.uint8).reshape(height_px, image.bytesPerLine())
    return argb[:, : width_px * 4].reshape(height_px, width_px, 4)[:, :, 3].copy()


# ------------------------------------------------------------- vs. raster

def shapes_match_the_raster():
    for example_id in ("differential", "filter"):
        info, d = example(example_id)
        ppm = float(info["preview"]["px_per_mm"])
        width_mm, height_mm = info["board"]["width_mm"], info["board"]["height_mm"]
        for name in [layer["name"] for layer in info["layers"]] + ["edge_cuts"]:
            w, h, coverage = board.render_preview_alpha(d, name, ppm)
            raster = np.frombuffer(coverage, np.uint8).reshape(h, w)
            vector = vector_alpha(board.preview_shapes(d, name), w, h, w / width_mm, h / height_mm, height_mm)
            a, b = raster > 127, vector > 127
            union = int((a | b).sum())
            assert union > 0, f"{example_id}/{name}: nothing drawn"
            # Allow antialiasing differences only.
            mismatch = int((a ^ b).sum()) / union
            assert mismatch < 0.03, f"{example_id}/{name}: {mismatch:.1%} of copper differs from the raster"


def clear_polarity_erases():
    gerber = WORK / "clear.gbr"
    gerber.write_text(
        "%FSLAX46Y46*%\n%MOMM*%\n%ADD10C,4.000000*%\n%LPD*%\n"
        "G36*\nX0Y0D02*\nX10000000Y0D01*\nX10000000Y10000000D01*\nX0Y10000000D01*\nX0Y0D01*\nG37*\n"
        "%LPC*%\nD10*\nX5000000Y5000000D03*\n%LPD*%\nX5000000Y5000000D03*\n"
        "%ADD11C,1.000000*%\n%LPC*%\nD11*\nX5000000Y5000000D03*\nM02*\n",
        encoding="ascii")
    shapes = gerber_shapes.layer_shapes(gerber, ((0.0, 0.0), (10.0, 10.0)))
    runs = sorted({(int(r), bool(k)) for r, k in shapes["index"][:, :2]})
    assert runs == [(0, True), (1, False), (2, True), (3, False)], runs
    alpha = vector_alpha(shapes, 100, 100, 10.0, 10.0, 10.0).astype(int)
    # Square, a 4 mm hole, a 4 mm pad back in it, a 1 mm hole in that pad.
    assert alpha[5, 5] >= 250 and alpha[50, 50] <= 5, (alpha[5, 5], alpha[50, 50])
    assert alpha[50, 50 + 15] >= 250, alpha[50, 65]  # 1.5 mm out: in the pad, past the small hole
    assert alpha[50, 50 + 25] >= 250 and alpha[50, 50 + 18] >= 250, "the pad must refill the big hole"
    assert alpha[50, 50 + 6:50 + 30].min() >= 250, "no ring left where the big hole was"


def arcs_stay_curves():
    gerber = WORK / "arc.gbr"
    gerber.write_text(
        "%FSLAX46Y46*%\n%MOMM*%\n%ADD10C,0.500000*%\n%ADD11C,6.000000*%\nD10*\n"
        "G75*\nX2000000Y5000000D02*\nG03X8000000Y5000000I3000000J0D01*\n"  # a half circle, counter-clockwise
        "D11*\nG01*\nX5000000Y5000000D03*\nM02*\n",
        encoding="ascii")
    shapes = gerber_shapes.layer_shapes(gerber, ((0.0, 0.0), (10.0, 10.0)))
    row = int(np.nonzero(shapes["index"][:, 2])[0][0])  # the trace
    offset = 0
    for i in range(row + 1):  # walk the serialised paths
        count = int(np.frombuffer(shapes["blob"][offset:offset + 4].tobytes(), "<i4")[0])
        elements = np.frombuffer(shapes["blob"][offset + 4:offset + 4 + 20 * count].tobytes(), gerber_shapes.ELEMENT)
        offset += 4 + 20 * count + 8
    assert list(elements["type"]) == [0, 2, 3, 3, 2, 3, 3], f"a half circle should be two cubics: {elements['type']}"
    decoded = decode_shapes(shapes)
    stroke = decoded[row]
    curves = sum(1 for i in range(stroke.path.elementCount()) if stroke.path.elementAt(i).isCurveTo())
    assert curves > 0, "the trace's outline lost its curves"
    # Counter-clockwise from (2, 5) to (8, 5) about (5, 5) passes through (5, 2), below the centre.
    assert abs(stroke.bounds.top() - (2 - 0.25)) < 0.3, stroke.bounds
    alpha = vector_alpha(shapes, 1000, 1000, 100.0, 100.0, 10.0)
    assert alpha[1000 - 200, 500] == 255, "the arc's lowest point is not drawn"  # (5, 2) mm
    assert alpha[500, 500] == 255 and alpha[500, 500 + 290] == 255, "the 6 mm flash is not filled"


def cache_round_trips_without_pickle():
    info, d = example("differential")
    name = info["layers"][0]["name"]
    first = board.preview_shapes(d, name)
    cached = list((d / "cache" / "preview").glob(f"{name}-shapes-*.npz"))
    assert len(cached) == 1, cached
    with np.load(cached[0], allow_pickle=False) as stored:
        assert np.array_equal(stored["index"], first["index"]) and np.array_equal(stored["blob"], first["blob"])
    t0 = time.perf_counter()
    again = board.preview_shapes(d, name)
    assert np.array_equal(again["blob"], first["blob"])
    assert time.perf_counter() - t0 < 2.0, "the second call should come from the cache"


def shapes_carry_their_nets():
    info, d = example("differential")
    top = board.preview_shapes(d, "F_Cu")
    plane = board.preview_shapes(d, "In1_Cu")
    assert {"/DIFF_P", "/DIFF_N", "GND"} <= set(top["nets"]), top["nets"]
    names = list(plane["nets"])
    filled = plane["index"][plane["index"][:, 2] == 0]  # pads and regions
    assert names == ["GND"] and (filled[:, 7] == 0).all(), "the plane (a region) carries its net too"
    nets = {n["name"]: n["layers"] for n in board.board_nets(d)}
    assert nets["/DIFF_P"] == ["F_Cu"] and set(nets["GND"]) == {"F_Cu", "In1_Cu", "In2_Cu", "B_Cu"}, nets


def canvas_colours_by_net():
    from EMShrimulator.app.views.layout import palette
    info, d = example("differential")
    info["nets"] = board.board_nets(d)
    canvas = BoardCanvas()
    canvas.raster_loader = lambda layer, ppm: board.render_preview_alpha(d, layer, ppm)
    canvas.shape_loader = lambda layer: board.preview_shapes(d, layer)
    canvas.visible_layers = {layer["name"]: layer["name"] == "F_Cu" for layer in info["layers"]}
    canvas.visible_layers["edge_cuts"] = False
    canvas.show_drills = False
    canvas.resize(640, 480)
    canvas.show()
    try:
        canvas.set_board(info, [])
        _settle(canvas)
        plain = canvas._layers["F_Cu"].toImage()
        # A point on the /DIFF_P trace: the middle of its longest straight run on F_Cu.
        trace = board.preview_shapes(d, "F_Cu")
        names = list(trace["nets"])
        rows = [r for r in trace["index"] if r[2] > 0 and int(r[7]) == names.index("/DIFF_P")]
        x0, y0, x1, y1 = max(rows, key=lambda r: (r[5] - r[3]) * (r[6] - r[4]))[3:7]
        hit = None
        for fx in np.linspace(0.1, 0.9, 41):
            for fy in np.linspace(0.1, 0.9, 41):
                found = canvas.net_at(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
                if found and found[0] == "/DIFF_P":
                    hit = (x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
                    break
            if hit:
                break
        assert hit, "no point of the /DIFF_P trace was found under the cursor"
        canvas.set_color_by("net")
        _settle(canvas)
        styled = canvas._layers["F_Cu"].toImage()
        assert canvas._styled.get("F_Cu") == canvas._style_key(), "the raster coloured by net never arrived"
        px, py = int(hit[0] * canvas.base_ppm), int((canvas.height_mm - hit[1]) * canvas.base_ppm)
        want = QColor(palette.net_colors(n["name"] for n in info["nets"])["/DIFF_P"])
        got, was = styled.pixelColor(px, py), plain.pixelColor(px, py)
        assert max(abs(got.red() - want.red()), abs(got.green() - want.green()), abs(got.blue() - want.blue())) < 40, \
            (got.name(), want.name())
        assert was.name() != got.name(), "colouring by net changed nothing"
        canvas.set_highlight("GND")
        _settle(canvas)
        faded = canvas._layers["F_Cu"].toImage().pixelColor(px, py)
        assert faded.name() != got.name(), "a highlight should fade the other nets"
        canvas.set_highlight(None)
        canvas.set_color_by("layer")
        assert canvas._layers["F_Cu"].toImage().pixelColor(px, py).name() == was.name(), "back to the layer colour"
    finally:
        _close(canvas)


def truncated_shapes_are_refused():
    info, d = example("differential")
    shapes = board.preview_shapes(d, info["layers"][0]["name"])
    try:
        decode_shapes({"index": shapes["index"], "blob": shapes["blob"][:-9]})
    except ValueError:
        return
    raise AssertionError("a truncated blob decoded without complaint")


# ---------------------------------------------------------------- canvas

def _settle(canvas, timeout=60.0):
    deadline = time.monotonic() + timeout
    while canvas._inflight and time.monotonic() < deadline:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)
    assert not canvas._inflight, f"still loading: {canvas._inflight}"
    app.processEvents()


def _mixed_fraction(image: QImage, colors) -> float:
    """Share of pixels that are none of `colors`: antialiased edges, or blur."""
    rgb = _pixels(image)
    pure = np.zeros(rgb.shape[:2], bool)
    for color in colors:
        pure |= np.abs(rgb - np.array([color.blue(), color.green(), color.red()])).max(axis=2) <= 6
    return float(1.0 - pure.mean())


def _pixels(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format_RGB32)
    rgb = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rgb[:, : image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, :3].astype(int)


def _zoomed_canvas():
    """The differential board's top layer zoomed 16x past its raster onto a copper edge. Returns the canvas
    and the two colours a sharp picture is made of."""
    info, d = example("differential")
    canvas = BoardCanvas()
    canvas.raster_loader = lambda layer, ppm: board.render_preview_alpha(d, layer, ppm)
    canvas.shape_loader = lambda layer: board.preview_shapes(d, layer)
    top = info["layers"][0]["name"]
    canvas.visible_layers = {layer["name"]: layer["name"] == top for layer in info["layers"]}
    canvas.visible_layers["edge_cuts"] = False
    canvas.show_drills = False
    canvas.resize(640, 480)
    canvas.show()
    canvas.set_board(info, board.drill_holes(d))
    _settle(canvas)
    assert set(canvas._shapes) == set(canvas._layers), "every layer should have its shapes"
    canvas.fit()
    app.processEvents()
    assert not canvas._sharp(), "the fitted board should be drawn from its rasters"

    mask = canvas._shown[top][1]
    coverage = np.frombuffer(mask.constBits(), np.uint8).reshape(mask.height(), mask.bytesPerLine())
    copper_px = coverage[:, : mask.width()] > 127
    rows, cols = np.nonzero(copper_px[:, 1:] != copper_px[:, :-1])
    cy, cx = mask.height() / 2, mask.width() / 2
    nearest = int(np.argmin((rows - cy) ** 2 + (cols - cx) ** 2))
    factor = 16.0 / (canvas.zoom() * canvas.devicePixelRatioF())
    canvas.scale(factor, factor)
    canvas.centerOn(float(cols[nearest]) + 1.0, float(rows[nearest]) + 0.5)
    app.processEvents()
    assert canvas._sharp()

    from EMShrimulator.app import theme
    c = theme.colors()
    backing, copper = QColor(c["sunken"]), QColor(info["layers"][0]["color"])
    opacity = canvas._opacity(top)
    over = QColor.fromRgbF(*(opacity * a + (1 - opacity) * b
                             for a, b in zip(copper.getRgbF()[:3], backing.getRgbF()[:3])))
    return canvas, (backing, over)


def _close(canvas) -> None:
    canvas.shutdown()
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settled_grab(canvas) -> QImage:
    """The view once any off-thread frame it asked for has been drawn."""
    canvas._redraw_board()
    image = canvas.viewport().grab().toImage()
    deadline = time.monotonic() + 30
    while (canvas._frame_queued or canvas._frame_busy) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if canvas._frame is not None:
        canvas._redraw_board()
        image = canvas.viewport().grab().toImage()
    return image


def canvas_stays_sharp_zoomed_in():
    canvas, colors = _zoomed_canvas()
    try:
        sharp = _settled_grab(canvas)
        canvas._redraw_board()
        t0 = time.perf_counter()
        canvas.viewport().grab()
        frame_ms = (time.perf_counter() - t0) * 1000
        shapes, canvas._shapes = canvas._shapes, {}  # what it drew before: the raster, stretched
        canvas._redraw_board()
        blurred = canvas.viewport().grab().toImage()
        canvas._shapes = shapes

        mixed_sharp, mixed_blur = _mixed_fraction(sharp, colors), _mixed_fraction(blurred, colors)
        print(f"   16x: in-between pixels {mixed_sharp:.2%} drawn from shapes, {mixed_blur:.2%} from the raster;"
              f" a repaint {frame_ms:.0f} ms")
        assert mixed_sharp < mixed_blur / 4, (mixed_sharp, mixed_blur)
    finally:
        _close(canvas)


def slow_copper_is_drawn_off_the_ui_thread():
    canvas, colors = _zoomed_canvas()
    try:
        canvas._redraw_board()
        in_place = canvas.viewport().grab().toImage()

        # Simulate a slow in-place draw: rasters first, and a frame requested.
        canvas._vector_ms = 1e9
        canvas._redraw_board()
        first = canvas.viewport().grab().toImage()
        assert _mixed_fraction(first, colors) > 4 * _mixed_fraction(in_place, colors), \
            "until the frame comes the rasters should stand in"
        deadline = time.monotonic() + 30
        while (canvas._frame is None or canvas._frame_busy) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert canvas._frame is not None, "no frame was drawn off the UI thread"

        canvas._vector_ms = 1e9  # keep showing the frame rather than drawing in place again
        canvas._redraw_board()
        framed = canvas.viewport().grab().toImage()
        app.processEvents()
        assert not canvas._frame_busy, "a frame that covers the view should not ask for another"
        # Must land on the same device pixels as drawing in place.
        diff = np.abs(_pixels(framed) - _pixels(in_place))
        assert diff.max() <= 3, f"the frame differs from drawing in place by up to {diff.max()} levels"
    finally:
        _close(canvas)


if __name__ == "__main__":
    try:
        check("shapes: match the resvg raster on every layer of two boards", shapes_match_the_raster)
        check("shapes: clear polarity erases what came before, dark refills", clear_polarity_erases)
        check("shapes: arcs and round flashes stay curves", arcs_stay_curves)
        check("shapes: cache round-trips as plain arrays", cache_round_trips_without_pickle)
        check("shapes: a truncated blob is refused", truncated_shapes_are_refused)
        check("shapes: every path keeps its X2 net, planes included", shapes_carry_their_nets)
        check("canvas: colour by net, the net under the cursor, a highlight fades the rest", canvas_colours_by_net)
        check("canvas: zoomed 16x the copper stays sharp", canvas_stays_sharp_zoomed_in)
        check("canvas: slow copper comes from a frame drawn off the UI thread, pixel for pixel",
              slow_copper_is_drawn_off_the_ui_thread)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    failed = 0
    for status, label, detail in results:
        print(f"{status}  {label}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
            failed += 1
    raise SystemExit(1 if failed else 0)
