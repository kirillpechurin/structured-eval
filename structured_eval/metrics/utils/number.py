"""Lenient numeric parsing shared by the numeric field metrics."""

from __future__ import annotations

import re
from typing import Any

# Everything that is not part of a (possibly scientific) number. Kept: digits,
# decimal point, signs, and the exponent marker e/E, so float() parses
# scientific notation ("1e3" → 1000.0, "1.5e-3" → 0.0015).
_NON_NUMERIC = re.compile(r"[^0-9eE.+\-]")


def parse_number(value: Any) -> float | None:
    """Coerce ``value`` to a float, or ``None`` if it isn't cleanly numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    negative = False
    # Accounting notation: "(123)" means -123.
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
        negative = True

    text = _NON_NUMERIC.sub("", text)
    if text in ("", "-", ".", "-."):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number
