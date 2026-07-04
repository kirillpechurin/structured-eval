from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


def _member(value: Any) -> Any:
    """A hashable, comparison-stable set key for one element.

    Scalars are used as-is; an unhashable element (dict/list) is keyed by its
    canonical JSON so set membership still works without a TypeError.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


class ArrayJaccardSimilarity(ArrayMetric):
    """Set-overlap (Jaccard) similarity for arrays, order- and count-insensitive.

    ``|A ∩ B| / |A ∪ B|`` over the two lists treated as **sets** (duplicates
    collapse, order is ignored):

    - ``1.0`` when the sets are identical (both empty → vacuously ``1.0``);
    - ``0.0`` when there is no overlap (or exactly one side is empty);
    - a value in ``(0, 1)`` otherwise.

    Built for arrays of scalars — tags, labels, categories. Membership is exact
    equality (no partial credit); for value-aware element matching use the
    aligned ``Array*`` P/R/F1 metrics instead.
    """

    name = "array_jaccard_similarity"

    def compute(self, node: ArrayNode) -> float:
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
        a = self._to_set(actual)
        e = self._to_set(expected)

        if not a and not e:
            return 1.0
        if not a or not e:
            return 0.0

        return len(a & e) / len(a | e)

    def _to_set(self, value: Any) -> set[Any]:
        """Convert a value to a set of hashable members."""
        if value is None:
            return set()
        if isinstance(value, (set, list, tuple)):
            return {_member(item) for item in value}
        return {_member(value)}
