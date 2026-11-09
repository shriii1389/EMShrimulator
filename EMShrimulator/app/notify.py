"""App-wide toasts (`notify`) and `guarded` for showing core errors instead of tracebacks."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from EMShrimulator.core.errors import CoreError, EditError, ValidationFailed

Sink = Callable[[str, str], None]
ValidationSink = Callable[[ValidationFailed], None]

_sink: Optional[Sink] = None
_validation_sink: Optional[ValidationSink] = None

EXPECTED = (CoreError, EditError)


def install(sink: Sink, validation_sink: Optional[ValidationSink] = None) -> None:
    global _sink, _validation_sink
    _sink = sink
    _validation_sink = validation_sink


def notify(message: str, kind: str = "info") -> None:
    if _sink is not None:
        _sink(message, kind)
    else:
        print(f"[{kind}] {message}")


def guarded(action: Callable[..., Any], *args, **kwargs) -> Any:
    """Run a user action; report an expected failure instead of raising it."""
    try:
        return action(*args, **kwargs)
    except ValidationFailed as exc:
        notify(exc.message, "error")
        if _validation_sink is not None:
            _validation_sink(exc)
    except EXPECTED as exc:
        notify(str(exc), "error")
    except Exception as exc:  # noqa: BLE001 - an unexpected bug still must not kill the app
        traceback.print_exc()
        notify(f"Unexpected error: {exc}", "error")
    return None
