from __future__ import annotations

from typing import Any

from structured_eval.metrics.protocol import FieldMetric


class ExactMatch(FieldMetric):
    """Strict equality: ``actual == expected`` → 1.0, else 0.0.

    The default field comparison and the default match criterion for object
    and array metrics.
    """

    name = "exact_match"

    def score(self, actual: Any, expected: Any) -> float:
        return 1.0 if actual == expected else 0.0
