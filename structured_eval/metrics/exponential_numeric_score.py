from __future__ import annotations

import math
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null
from structured_eval.metrics.utils.number import parse_number


class ExponentialNumericScore(FieldMetric):
    """Exponentially decaying similarity for numeric fields.

    The score is computed as::

        exp(-abs(actual - expected) / scale)

    yielding:

    - ``1.0`` for an exact match;
    - a smooth exponential decay as the absolute error increases;
    - values always in the range ``(0.0, 1.0]``.

    The ``scale`` parameter controls how quickly the score decreases. Larger
    values make the metric more tolerant to numeric differences. Unlike the
    ratio-based :class:`NumericCloseness`, the decay is on the **absolute**
    error, so it is scale-aware — pick ``scale`` to match the field's units.

    Values are read with the same lenient parser as :class:`Numeric` /
    :class:`NumericCloseness`, so numeric strings are graded too. The metric
    applies **only to numbers**: if either side isn't numeric (``None``, a
    non-numeric string, or a ``bool`` — ``True`` is not ``1``) the score is
    ``0.0``. Two ``None``s are the exception — they agree (``1.0``; see
    ``metrics.utils.null``).
    """

    name = "exponential_numeric_score"

    def __init__(self, scale: float = 1.0, name: str | None = None) -> None:
        super().__init__(name=name)
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        self.scale = scale

    def score(self, actual: Any, expected: Any) -> float:
        if both_null(actual, expected):
            return 1.0
        a = parse_number(actual)
        e = parse_number(expected)
        if a is None or e is None:
            return 0.0
        return math.exp(-abs(a - e) / self.scale)
