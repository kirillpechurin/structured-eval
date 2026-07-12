"""The ``(None, None) → 1.0`` rule shared by the comparison field metrics.

The schemas under evaluation ask for ``null`` whenever a value is absent, so a
null expectation met by a null answer is a *correct* answer. Without this gate a
metric's type check (str / number / date) would reject the pair and score that
right answer ``0.0``. Only ``None`` counts as null — an empty string, an empty
list or a missing key are values, and are graded as such.

A one-sided ``None`` stays a mismatch: a value was expected and nothing came
back, or nothing was expected and a value was invented.
"""

from __future__ import annotations

from typing import Any


def both_null(actual: Any, expected: Any) -> bool:
    """True when neither side has a value — the two agree."""
    return actual is None and expected is None
