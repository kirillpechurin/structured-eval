from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.number import parse_number


class NumericCloseness(FieldMetric):
    """Graded numeric similarity in ``[0, 1]`` (not a pass/fail tolerance).

    ``1 - |actual - expected| / max(|actual|, |expected|)`` — the ratio of the
    smaller magnitude to the larger (``min/max`` for same-sign values): equal
    values score 1.0, opposite signs trend toward 0.0, and ``0/0`` is 1.0.
    Unlike :class:`Numeric` (a hard 0/1 verdict against a tolerance), this yields
    a continuous score, making it the default element scorer for numbers under
    the Hungarian array aligner where a graded cost matrix matters.

    Values are parsed with the shared lenient numeric parser (same as
    :class:`Numeric`), so numeric strings are graded too. The metric applies
    **only to numbers**: if either side isn't numeric (``None``, a non-numeric
    string, or a ``bool`` — ``True`` is not ``1``) the score is 0.0.
    """

    name = "numeric_closeness"

    def score(self, actual: Any, expected: Any) -> float:
        a = parse_number(actual)
        e = parse_number(expected)
        if a is None or e is None:
            return 0.0
        if a == e:
            return 1.0
        denom = max(abs(a), abs(e))
        return max(0.0, 1.0 - abs(a - e) / denom) if denom else 1.0
