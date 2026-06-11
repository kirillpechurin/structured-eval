from __future__ import annotations

from typing import Any, Literal

from structured_eval.matchers.protocol import MatcherBase

Mode = Literal["relative", "absolute"]


class NumericMatcher(MatcherBase):
    """Numeric equality within a tolerance band → 1.0, otherwise 0.0.

    ``relative`` mode measures the deviation as ``|a - e| / |e|``; ``absolute``
    measures ``|a - e|``. A tolerance of 0 means exact numeric equality.
    Non-numeric values yield 0.0.
    """

    name = "NUMERIC"

    def __init__(self, tolerance: float = 0.01, mode: Mode = "relative"):
        self.tolerance = tolerance
        self.mode = mode

    def similarity(self, actual: Any, expected: Any) -> float:
        try:
            a, e = float(actual), float(expected)
        except (TypeError, ValueError):
            return 0.0

        if self.mode == "relative":
            if e == 0:
                deviation = 0.0 if a == 0 else float("inf")
            else:
                deviation = abs(a - e) / abs(e)
        else:
            deviation = abs(a - e)

        return 1.0 if deviation <= self.tolerance else 0.0
