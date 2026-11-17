"""PyInstaller entry point for gems-worker(.exe). See core/workers.py."""
from EMShrimulator.core.workers import main

if __name__ == "__main__":
    raise SystemExit(main())
