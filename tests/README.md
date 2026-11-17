# Tests

Run everything from the repo root:

```bash
./gems.sh test        # Linux
.\gems.ps1 test       # Windows (PowerShell)
```

There are two kinds of file here. Some are normal pytest files, and the rest
are scripts that print a PASS/FAIL line per check and exit non-zero if any
fail. Mostly that's because they drive native code (openEMS, VTK, the
vendored gerber2ems) in ways that are easier to follow as a script.
`gems.sh test` and `gems.ps1 test` work out which is which and run both.

To run one on its own:

```bash
export PYTHONPATH=$PWD LD_LIBRARY_PATH=~/opt/openEMS/lib64:~/opt/openEMS/lib
.venv/bin/python -m pytest -q tests/test_reports.py   # a pytest file
.venv/bin/python tests/test_sheets.py                 # a script
```

On Windows that's `.venv\Scripts\python.exe` with `$env:PYTHONPATH = (Get-Location).Path`.

`test_modeler.py` and `test_sheets.py` run short real solves, so they need a
working openEMS install and take a while. The rest don't solve anything.

## What's covered

| File | Checks |
| --- | --- |
| `test_modeler.py` | Revolves against CSXCAD, a 50/100 Ω two-port that has to come out reciprocal, a dipole's directivity, WR-90 and coax lines end to end |
| `test_sheets.py` | Sheets and sheet ports, ending with a real solve. openEMS happily accepts some broken setups and only fails at run time, so this reads its actual output |
| `test_reports.py` | Every report against closed-form answers: patterns, polarisations, axial ratio, Y = Z⁻¹, TDR, near-field planes |
| `test_reports_ui.py` | Every New report preset in every display, plus adding, editing and removing traces through the UI |
| `test_touchstone.py` | Touchstone read/write |
| `test_modeler_workspace.py` | The modeler UI and every template validating |
| `test_snapping.py`, `test_coordinate_systems.py`, `test_coordinate_system_ui.py` | Snapping, measuring and coordinate systems |
| `test_assign_port_ui.py`, `test_define_variable.py`, `test_viewport_clicks.py` | Port assignment, variables and viewport clicks |
| `test_layout_ports.py`, `test_board_shapes.py` | Layout ports, and the 2D canvas copper against the rendered rasters |
| `test_copper_edit.py` | Editing copper and writing it back to Gerbers with the nets intact |
| `test_padstacks.py` | Drill files to padstacks and vias, and editing them |
| `test_diff_pairs.py`, `test_layout_pairs_ui.py` | Differential pairs and mixed-mode S |
| `test_odbpp.py` | ODB++ import. Set `GEMS_ODB_SAMPLE` to Siemens's `designodb_rigidflex.tgz` to try a real board too |
| `test_project_archive.py` | Exporting and importing projects, including unsafe archive paths |

One gotcha if you write a new solver check: openEMS prints from C++ straight
to file descriptor 1, so `contextlib.redirect_stdout` won't catch it. Capture
at the fd level instead, like `test_sheets.py` does.
