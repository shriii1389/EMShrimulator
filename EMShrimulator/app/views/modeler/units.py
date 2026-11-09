"""Display-unit conversion for dimension fields. Stored values stay in mm/Hz/deg."""
from __future__ import annotations

from EMShrimulator.core.em.model import DisplayUnits
from EMShrimulator.core.em.params import FREQUENCY_UNITS, LENGTH_UNITS

QUANTITIES = ("length", "frequency", "angle", "plain")


def factor(quantity: str, units: DisplayUnits) -> float:
    """How many base units one display unit is worth."""
    if quantity == "length":
        return LENGTH_UNITS.get(units.length, 1.0)
    if quantity == "frequency":
        return FREQUENCY_UNITS.get(units.frequency, 1.0)
    return 1.0


def label(quantity: str, units: DisplayUnits) -> str:
    if quantity == "length":
        return units.length
    if quantity == "frequency":
        return units.frequency
    if quantity == "angle":
        return "°"
    return ""
