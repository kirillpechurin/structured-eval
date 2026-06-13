from __future__ import annotations

from typing import Any, Literal

from structured_eval.metrics.protocol import FieldMetric

Mode = Literal["relative", "absolute"]


class Numeric(FieldMetric):
    """Numeric equality within a tolerance band → 1.0, otherwise 0.0.

    ``relative`` mode measures deviation as ``|a - e| / |e|``; ``absolute``
    measures ``|a - e|``. A tolerance of 0 means exact numeric equality.
    Non-numeric values yield 0.0.
    """

    name = "numeric"

    def __init__(self, tolerance: float = 0.01, mode: Mode = "relative"):
        self.tolerance = tolerance
        self.mode = mode

    def score(self, actual: Any, expected: Any) -> float:
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
