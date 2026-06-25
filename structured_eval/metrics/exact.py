from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric


class ExactMatch(FieldMetric):
    """Strict equality: ``actual == expected`` → 1.0, else 0.0.

    The default scalar comparison, and the default key comparison in ``by_key``
    array alignment. It does *not* score whole objects/arrays: object metrics
    read each child's representative, and array alignment defaults are
    type-aware — ExactMatch only ever touches a dict/list through the
    value-level ``score`` path.
    """

    name = "exact_match"

    def score(self, actual: Any, expected: Any) -> float:
        return 1.0 if actual == expected else 0.0
