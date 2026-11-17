"""Reports through the window: every New report preset in every 2D display, from a synthetic run, plus
the 3D Layout's reports. Off screen, so 3D displays are skipped.

Run with `.venv/bin/python tests/test_reports_ui.py`; the exit code is the verdict.
"""
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="gems-reports-ui-"))
os.environ["GEMS_PROJECTS_DIR"] = str(WORK / "projects")
os.environ["GEMS_TEMP_DIR"] = str(WORK / "tmp")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"))


def synthetic_run(folder: Path):
    """A one-port run of a short dipole: Touchstone, far field at three frequencies, a field dump."""
    import numpy as np

    from EMShrimulator.core.em import farfield as ff
    from EMShrimulator.core.em.results import read_results
    from EMShrimulator.core.em.touchstone import write_touchstone

    freqs = np.linspace(0.05e9, 4e9, 80)
    s11 = 0.3 * np.exp(-1j * freqs / 1e9)
    data = read_results(write_touchstone(freqs, {1: {1: s11}}, [1], 50.0, folder / "dipole"), epsilon_eff=1.0)
    theta, phi = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
    e_theta = np.sin(np.radians(theta))[:, None] * np.ones((1, phi.size)) + 0j
    e_phi = 0.1j * e_theta
    entries = []
    for f in (2e9, 2.5e9, 3e9):
        u = (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2) / (2 * ff.ETA0)
        prad = float((u * ff.sphere_weights(theta, phi)).sum())
        entry = ff.summarise(f, theta, phi, 4 * np.pi * u / prad)
        entry.update(radiated_power_w=prad, accepted_power_w=prad / 0.9, incident_power_w=prad / 0.8,
                     efficiency=0.9, gain_dbi=entry["dmax_dbi"] - 0.46, realized_gain_dbi=entry["dmax_dbi"] - 0.97,
                     center_mm=[0.0, 0.0, 0.0])
        entries.append(entry)
    fields = {"frequency_hz": np.array([2e9, 2.5e9, 3e9]), "theta_deg": theta, "phi_deg": phi,
              "radius_m": np.array([1.0]), "e_theta": np.repeat(e_theta[None], 3, 0).astype(np.complex64),
              "e_phi": np.repeat(e_phi[None], 3, 0).astype(np.complex64)}
    x, y, z = np.linspace(-10, 10, 21), np.linspace(-5, 5, 11), np.array([0.0])
    field = np.zeros((2, 3, 1, y.size, x.size), dtype=np.complex64)
    field[:, 2] = 1.0 / (1.0 + x[None, None, :] ** 2 + y[None, :, None] ** 2)
    near = {"x_mm": x, "y_mm": y, "z_mm": z, "frequency_hz": np.array([2e9, 3e9]), "field": field}
    data.update(available=True, nearfield=[{"dump": "probe", "port": 1}],
                summary={"port_powers": {"freq_hz": freqs.tolist(), "1": {"incident": [4.0] * 80,
                                                                         "accepted": [3.0] * 80}}})
    return data, {"frequencies": entries, "fields": fields}, near


def main():
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, Signal
    from PySide6.QtWidgets import QApplication

    QApplication.setOrganizationName("EMShrimulator-tests")  # keep QSettings writes out of the user's
    app = QApplication.instance() or QApplication([])
    QSettings().setValue("layout/color_by", "layer")
    from EMShrimulator.app import commands as cmd
    from EMShrimulator.app.session import ModelerSession
    from EMShrimulator.app.views.results.plot_page import PlotPage
    from EMShrimulator.core.em.edits import DISPLAYS, PLOT_MENU, find_plot
    from EMShrimulator.core.em.model import DumpKind, FieldDump, ModelerProject

    def pump(seconds=0.0, until=None, timeout=60.0):
        start = time.monotonic()
        while True:
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            done = until() if until else True
            if done and time.monotonic() - start >= seconds:
                return
            if time.monotonic() - start > timeout:
                raise TimeoutError("the window did not settle")
            time.sleep(0.01)

    class Source(QObject):
        loaded = Signal()

    data, farfield, near = synthetic_run(WORK)
    source = Source()
    source.data, source.farfield = data, farfield
    session = ModelerSession("t", ModelerProject(name="t", dumps=[
        FieldDump(name="probe", kind=DumpKind.E_FREQ, p1=(-10, -5, 0), p2=(10, 5, 0), frequencies=[2e9, 3e9])]))
    loads = []

    def near_loader(dump, port):
        loads.append((dump, port))
        return near

    page = PlotPage(session, source, near_loader=near_loader)
    page.resize(1300, 800)
    page.show()
    flat = ("rect3d", "spherical", "cylindrical", "overlay")

    def every_report_in_every_2d_display():
        drawn = 0
        for _group, presets in PLOT_MENU:
            for preset in presets:
                plot_id = session.push(cmd.AddPlot(session, preset))
                plot = find_plot(session.project, plot_id)
                if plot.display in flat:  # no OpenGL off screen: start these on a 2D display
                    session.push(cmd.SetPlotField(session, plot_id, "display", DISPLAYS[plot.kind][0]))
                page.show_plot(plot_id)
                if preset == "cross_pol":  # it asks for the co-polarisation rather than guess one
                    assert page.stack.currentWidget() is page.message and "co-polarisation" in page.message.text()
                    session.push(cmd.SetPlotField(session, plot_id, "component", "theta"))
                for display in DISPLAYS[find_plot(session.project, plot_id).kind]:
                    if display in flat:
                        continue
                    if find_plot(session.project, plot_id).display != display:
                        session.push(cmd.SetPlotField(session, plot_id, "display", display))
                    pump(0.05, until=lambda: page._near_pending is None)
                    current = page.stack.currentWidget()
                    assert current in (page.chart, page.table), (preset, display, page.message.text())
                    assert page._export[0] and page._export[1], (preset, display, "nothing to export")
                    drawn += 1
        assert drawn > 50, drawn
        assert loads == [("probe", 1)], loads  # read once, then kept

    def options_follow_the_report():
        plot_id = session.push(cmd.AddPlot(session, "pattern_polar"))
        page.show_plot(plot_id)
        assert page.options.isRowVisible(page.cuts) and not page.options.isRowVisible(page.rise)
        assert page.add_category.value() in ("gain", "directivity") and not page.add_form.isRowVisible(
            page.add_direction)  # a pattern trace is cut, not read at one direction
        page.cuts.setText("0, 45, 90")
        page.cuts.editingFinished.emit()
        assert find_plot(session.project, plot_id).cuts == [0.0, 45.0, 90.0]
        assert len(page.chart._lines) > 3
        page.add_category.set_value("axial_ratio")
        page._fill_add_box()
        page.add_trace()
        assert [c.category for c in find_plot(session.project, plot_id).curves] == ["directivity", "axial_ratio"]
        session.undo_stack.undo()
        assert find_plot(session.project, plot_id).curves == []
        far_id = session.push(cmd.AddPlot(session, "gain_freq"))
        page.show_plot(far_id)
        page.add_category.set_value("gain")
        page._fill_add_box()
        assert page.add_form.isRowVisible(page.add_direction)  # against frequency, it is read at (θ, φ)
        tdr_id = session.push(cmd.AddPlot(session, "tdr"))
        page.show_plot(tdr_id)
        page.rise.setText("100")
        page.rise.editingFinished.emit()
        assert find_plot(session.project, tdr_id).rise_time_ps == 100.0
        page.rise.setText("")
        page.rise.editingFinished.emit()
        assert find_plot(session.project, tdr_id).rise_time_ps is None

    def a_near_field_is_for_one_watt_incident():
        plot_id = session.push(cmd.AddPlot(session, "nearfield"))
        session.push(cmd.SetPlotField(session, plot_id, "db", False))
        page.show_plot(plot_id)
        pump(0.05, until=lambda: page._near_pending is None)
        assert "for 1 W incident" in page.summary.text()
        peak = max(float(row[2]) for row in page._export[1])
        assert abs(peak - 0.5) < 1e-6, peak  # 1 V/m at openEMS's level, 4 W incident: 1/√4

    def add_trace_puts_z11_beside_s11():
        plot_id = session.push(cmd.AddPlot(session, "s_db"))
        page.show_plot(plot_id)
        assert [page.traces.item(i).text() for i in range(page.traces.count())] == ["dB(S11)"]
        page.add_category.set_value("z")
        page._fill_add_box()
        page.add_term.set_value("Z11")
        page.add_function.set_value("re")
        page.add_trace()
        plot = find_plot(session.project, plot_id)
        assert [(c.category, c.term, c.function) for c in plot.curves] == [("s", "S11", "db"), ("z", "Z11", "re")]
        assert page.chart._has_y2() and page.chart._y2_label.startswith("Impedance")
        page.traces.setCurrentRow(1)  # its settings come up in the box, to change in place
        assert page.add_term.value() == "Z11" and page.add_function.value() == "re"
        page.add_function.set_value("im")
        page.update_trace()
        assert [c.function for c in find_plot(session.project, plot_id).curves] == ["db", "im"]
        page.traces.item(0).setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Unchecked)
        assert not find_plot(session.project, plot_id).curves[0].visible
        assert page.stack.currentWidget() is page.chart  # Z11 alone, still drawn
        page.traces.setCurrentRow(1)
        page.remove_trace()
        assert [c.term for c in find_plot(session.project, plot_id).curves] == ["S11"]
        assert "hidden" in page.message.text() and page.stack.currentWidget() is page.message
        page.traces.item(0).setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Checked)
        session.push(cmd.SetPlotField(session, plot_id, "display", "smith"))
        assert page.stack.currentWidget() is page.chart

    def an_elevation_pattern_at_every_frequency():
        plot_id = session.push(cmd.AddPlot(session, "elevation_rect"))
        page.show_plot(plot_id)
        page.all_frequencies.click()
        assert find_plot(session.project, plot_id).all_frequencies
        labels = [line.label for line in page.chart._lines if line.label and not line.guide]
        assert len(labels) == 2 * 3 and any("φ = 90°" in label and "3 GHz" in label for label in labels), labels

    check("Every report preset draws in every 2D display, and exports", every_report_in_every_2d_display)
    check("Add trace puts Z11 beside S11, on a second axis; traces hide and remove", add_trace_puts_z11_beside_s11)
    check("An elevation pattern overlays every far-field frequency", an_elevation_pattern_at_every_frequency)
    check("The options shown follow the report; edits are undoable", options_follow_the_report)
    check("A near field is scaled to 1 W incident", a_near_field_is_for_one_watt_incident)
    page.shutdown()
    page.close()

    def the_layout_makes_reports_too():
        from EMShrimulator.app.session import LayoutSession
        from EMShrimulator.app.views.layout.layout_view import LayoutView
        from EMShrimulator.core import project as storage
        from EMShrimulator.core.layout import templates

        name = "board"
        d = storage.projects_dir() / name
        d.mkdir(parents=True)
        project = templates.build("differential", name, d)
        (d / "kind.txt").write_text("layout")
        storage.save_project(name, project)
        layout = LayoutSession(name, storage.load_project(name))
        view = LayoutView(layout)
        view.resize(1300, 800)
        view.show()
        pump(0.3)
        menu = view.fill_report_menu(__import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu())
        labels = [a.text() for a in menu.actions()]
        assert "TDR impedance" in labels and not any(a.menu() for a in menu.actions()), labels  # no far/near fields
        view.new_plot("y")
        pump(0.1)
        (plot,) = layout.project.plots
        assert view._page_key == "plot" and ("plot", plot.id) in set(view.tree.keys())
        assert view.pages["plot"].subtitle.text().startswith("Solution data · Rectangular")
        storage.save_project(name, layout.project)
        assert storage.load_project(name).plots[0].kind == "y"
        view.delete_plot(plot.id)
        assert view._page_key == "results" and not layout.project.plots
        view.shutdown()
        view.close()
        pump(0.2)

    check("The 3D Layout makes, lists, saves and deletes reports", the_layout_makes_reports_too)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print()
    for status, label, detail in results:
        print(f"  {status}  {label}")
        if detail:
            print(f"        {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"\n{len(results) - fails}/{len(results)} passed")
    raise SystemExit(1 if fails or not results else 0)
