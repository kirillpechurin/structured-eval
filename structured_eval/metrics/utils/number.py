"""Lenient numeric parsing shared by the numeric field metrics."""

from __future__ import annotations

import re
from typing import Any

# Everything that is not part of a (possibly scientific) number. Kept: digits,
# decimal point, signs, and the exponent marker e/E, so float() parses
# scientific notation ("1e3" → 1000.0, "1.5e-3" → 0.0015).
_NON_NUMERIC = re.compile(r"[^0-9eE.+\-]")


def parse_number(value: Any) -> float | None:
    """Coerce `value` to a float, or `None` if it isn't cleanly numeric.

    A model writes numbers the way a document does, so the string forms are
    read too:

    - currency, thousands separators and stray text are stripped
      (`"$1,234.50"` → `1234.5`);
    - accounting parentheses mean a negative (`"(123)"` → `-123.0`);
    - scientific notation is kept intact (`"1e3"` → `1000.0`).

    A `bool` is rejected rather than read as `0`/`1`: `True` is an answer to a
    different question than the number one, and scoring it as `1` would hide
    a wrongly-typed field.

    Args:
        value: The value to read a number out of.

    Returns:
        The number, or `None` when nothing numeric is left to parse.
    """
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
