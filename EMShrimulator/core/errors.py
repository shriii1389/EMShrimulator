"""Exceptions raised by core services, with user-facing messages."""
from __future__ import annotations

from typing import Dict, List, Optional


class CoreError(Exception):
    """Base class for every expected, user-facing failure."""


class NotFound(CoreError):
    """A project, template, result or file that does not exist."""


class InvalidInput(CoreError):
    """A request that cannot be honoured as asked - a bad name, a wrong kind."""


class AlreadyExists(CoreError):
    """Creating something whose name is taken."""


class ValidationFailed(CoreError):
    """A design that must not be solved; `issues` uses the validator's shape."""

    def __init__(self, message: str, issues: Optional[List[Dict]] = None):
        super().__init__(message)
        self.message = message
        self.issues = issues or []


class EditError(ValueError):
    """An edit that cannot be applied, with a message for the user."""
