# How it works

A quick map of the code, for anyone who wants to change something. The code
is the real reference; this is just enough to find your way around.

## Two halves

```
EMShrimulator/
  core/      the engine. No Qt, runs headless.
    em/        3D Modeler: project schema, expressions, geometry, mesh, solve, results
    layout/    3D Layout: Gerber/ODB++ import, canvas data, ports, padstacks, solve
    services/  what the UI calls into
  app/       the desktop app (PySide6, VTK for 3D, QPainter for charts)
```

The one rule: `app` imports `core`, never the other way round. There's no
PySide6 import anywhere under `core/`. That's what lets the solver run in its
own process without loading Qt, and why the packaged `gems-worker.exe` is
built without it.

## Projects

A project is a folder under `projects\` (or `Documents\EMShrimulator Projects`
in the Windows build):

```
<project>/
  kind.txt        "modeler" or "layout"
  project.json    the whole design, a pydantic model saved as JSON
  jobs/           the full log of every run
  results/        Touchstone files, far field, sweeps   (modeler)
  fab/, ems/      board files and gerber2ems output     (layout)
```

Expressions like `Wp/2 + 40mil` are parsed and checked against a whitelist in
`core/em/params.py`. Nothing in a project file is ever passed to `eval`.

## Editing

Every edit is an undo command (`app/commands.py`). It deep-copies the project,
applies a plain function from `core/em/edits.py` or `core/layout/edits.py`, and
swaps the copy in. Undo swaps the old one back.

After each edit a single background thread re-runs validation, builds the
CSXCAD mesh preview and tessellates the geometry, so the 3D view always shows
what the solver will actually get. If you keep typing, older previews are
thrown away.

Gerber parsing and rasterising hold the GIL, so they run in a small process
pool instead of threads. Otherwise the UI would freeze on a big board.

## Running a solve

Press F5 and the project is saved, validated and handed to a separate solver
process. The UI never runs openEMS itself.

- Modeler: `core/em/run_worker.py` builds the CSXCAD model, runs openEMS once
  per excited port and writes a Touchstone file plus the far field.
- Layout: the board goes through the patched gerber2ems in
  `vendor/gerber2ems-main`. It's Antmicro's tool with lumped and circuit ports,
  padstacks, lossy substrates and an early stop once the S-parameters settle.
  Every change is marked `PATCHED (EMShrimulator Studio)` or
  `ADDED (EMShrimulator Studio)` in the source.

The worker writes everything to `jobs/<id>.log`, including progress lines
starting with `@@GEMS`. The UI just tails that file, which is why a run keeps
its log even if the app closes. When the job finishes, results load on a
background thread and the reports redraw.

## Reports

The maths behind every trace (S to Y and Z, VSWR, group delay, TDR, gain,
axial ratio and so on) lives in `core/em/reports.py`, with no UI code. The
drawing is in `app/views/results/`. `tests/test_reports.py` checks the maths
against closed-form answers.

## Tests

See [`tests/README.md`](../tests/README.md).
