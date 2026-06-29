from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.model.nodes.array_node import ArrayNode


class ArrayExactMatch(ArrayMetric):
    """Strict whole-array equality: identical lists → 1.0, else 0.0.

    Compares the raw ``actual`` / ``expected`` lists element-by-element,
    **order-sensitively** and recursively (nested dicts/lists are deep-compared).
    No alignment, no partial credit — the array as a whole is either right or
    wrong. Use it when element order is part of correctness; for set-style or
    value-aware scoring reach for :class:`ArrayJaccardSimilarity` or the
    aligned ``Array*`` P/R/F1 metrics instead.
    """

    name = "array_exact_match"

    def compute(self, node: ArrayNode) -> float:
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
        return 1.0 if self._array_equal(actual, expected) else 0.0

    def _array_equal(self, a: Any, b: Any) -> bool:
        """Strict order-sensitive array comparison."""
        if not (isinstance(a, list) and isinstance(b, list)):
            return False
        if len(a) != len(b):
            return False
        return all(self._deep_equal(x, y) for x, y in zip(a, b, strict=False))

    def _deep_equal(self, a: Any, b: Any) -> bool:
        """Shared recursive equality helper."""
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._deep_equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return self._array_equal(a, b)
        return bool(a == b)
