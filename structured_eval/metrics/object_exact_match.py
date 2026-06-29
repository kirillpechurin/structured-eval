from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric

if TYPE_CHECKING:
    from structured_eval.model.nodes.object_node import ObjectNode


class ObjectExactMatch(ObjectMetric):
    """Strict deep equality for objects: identical dicts → 1.0, else 0.0.

    Compares the raw ``actual`` / ``expected`` mappings recursively — same keys,
    and every value deep-equal (nested dicts and lists included). No partial
    credit and no coercion: the object as a whole is either right or wrong. For
    field-level partial credit use the aggregating ``Object*`` metrics
    (``ObjectAccuracy`` / ``ObjectF1`` …) instead.
    """

    name = "object_exact_match"

    def compute(self, node: ObjectNode) -> float:
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
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
