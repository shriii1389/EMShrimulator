"""The Qt desktop application. Depends on `EMShrimulator.core`; core never imports this."""
from __future__ import annotations


def _skip_pyside_feature_scan() -> None:
    """Skip PySide6's source scan of every later import (for __feature__, unused here); it slows startup."""
    try:
        import PySide6  # noqa: F401 - installs the hook being short-circuited
        from shibokensupport import feature
    except ImportError:  # pragma: no cover
        return
    if callable(getattr(feature, "_mod_uses_pyside", None)):
        feature._mod_uses_pyside = lambda _module: True


_skip_pyside_feature_scan()
