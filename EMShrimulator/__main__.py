"""`python -m EMShrimulator [--open "<project>"]` starts the desktop app."""
from __future__ import annotations

import sys


def main() -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    from EMShrimulator.app.application import run

    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
