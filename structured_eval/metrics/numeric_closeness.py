from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric


class NumericCloseness(FieldMetric):
    """Graded numeric similarity in ``[0, 1]`` (not a pass/fail tolerance).

    ``1 - |actual - expected| / max(|actual|, |expected|)`` — equal values score
    1.0, opposite signs trend toward 0.0, and ``0/0`` is 1.0. Unlike
    :class:`Numeric` (which returns a hard 0/1 verdict against a tolerance), this
    yields a continuous score, making it the default element scorer for numbers
    under the Hungarian array aligner where a graded cost matrix matters.

    Non-numeric (including ``bool``) values score 0.0 unless strictly equal.
    """

    name = "numeric_closeness"

    def score(self, actual: Any, expected: Any) -> float:
        a = self._to_number(actual)
        e = self._to_number(expected)
        if a is None or e is None:
            return 1.0 if actual == expected else 0.0
        if a == e:
            return 1.0
        denom = max(abs(a), abs(e))
        return max(0.0, 1.0 - abs(a - e) / denom) if denom else 1.0

    @staticmethod
    def _to_number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None
