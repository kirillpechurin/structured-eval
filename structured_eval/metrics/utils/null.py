"""The `(None, None) → 1.0` rule shared by the comparison field metrics."""

from __future__ import annotations

from typing import Any


def both_null(actual: Any, expected: Any) -> bool:
    """True when neither side has a value — the two agree."""
    return actual is None and expected is None
