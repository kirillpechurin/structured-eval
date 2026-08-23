"""The `object_exact_match` metric — strict deep equality for objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.object_node import ObjectNode


class ObjectExactMatch(ObjectMetric):
    """Strict deep equality for objects: identical dicts → 1.0, else 0.0.

    Compares the two mappings recursively — same keys, every value deep-equal.
    No partial credit and no coercion: the object as a whole is either right or
    wrong. For field-level partial credit use the aggregating `Object*` metrics
    instead.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ObjectExactMatch
        >>> from structured_eval.models import EvalConfig
        >>> ObjectExactMatch().score({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})
        1.0
        >>> ObjectExactMatch().score({"a": 1}, {"a": "1"})     # no coercion
        0.0
        >>> report = evaluate({"a": 1}, {"a": "1"},
        ...                   EvalConfig(metrics=[ObjectExactMatch()]))
        >>> float(report.metrics["object_exact_match"].representative())
        0.0
    """

    name = "object_exact_match"

    def compute(self, node: ObjectNode) -> float:
        """1.0 when this node's two mappings are identical, else 0.0."""
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when the two mappings are deep-equal, else 0.0."""
        return 1.0 if self._object_equal(actual, expected) else 0.0

    def _object_equal(self, a: Any, b: Any) -> bool:
        """Deep strict equality for JSON-like structures."""
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._object_equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(self._object_equal(x, y) for x, y in zip(a, b, strict=False))
        return bool(a == b)
