"""The `array_exact_match` metric — strict whole-array equality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayExactMatch(ArrayMetric):
    """Strict whole-array equality: identical lists → 1.0, else 0.0.

    Compares the raw `actual` / `expected` lists element-by-element,
    **order-sensitively** and recursively — nested dicts and lists are
    deep-compared. No alignment, no partial credit: the array as a whole is
    either right or wrong.

    Use it when element order is part of correctness; for set-style or
    value-aware scoring reach for `ArrayJaccardSimilarity` or the aligned
    `Array*` P/R/F1 metrics instead.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayExactMatch
        >>> from structured_eval.models import EvalConfig
        >>> ArrayExactMatch().score([1, 2], [1, 2])
        1.0
        >>> ArrayExactMatch().score([1, 2], [2, 1])     # order-sensitive
        0.0
        >>> report = evaluate({"tags": [2, 1]}, {"tags": [1, 2]},
        ...                   EvalConfig(metrics=[ArrayExactMatch()]))
        >>> float(report.metrics["array_exact_match"].representative())
        0.0
    """

    name = "array_exact_match"

    def compute(self, node: ArrayNode) -> float:
        """1.0 when this node's two lists are identical, else 0.0."""
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when the two lists are deep-equal in order, else 0.0."""
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
