"""The `array_jaccard_similarity` metric — set overlap, blind to order and counts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


def _member(value: Any) -> Any:
    """A hashable, comparison-stable set key for one element.

    Args:
        value: One array element.

    Returns:
        The value itself when hashable, else its canonical JSON, so that set
        membership works without a `TypeError`.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


class ArrayJaccardSimilarity(ArrayMetric):
    """Set-overlap (Jaccard) similarity for arrays, order- and count-insensitive.

    `|A ∩ B| / |A ∪ B|` over the two lists treated as **sets** (duplicates
    collapse, order is ignored):

    - `1.0` when the sets are identical (both empty → vacuously `1.0`);
    - `0.0` when there is no overlap (or exactly one side is empty);
    - a value in `(0, 1)` otherwise.

    Built for arrays of scalars — tags, labels, categories. Membership is exact
    equality, with no partial credit; for value-aware element matching use the
    aligned `Array*` P/R/F1 metrics instead.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayJaccardSimilarity
        >>> from structured_eval.models import EvalConfig
        >>> ArrayJaccardSimilarity().score([1, 2, 3], [2, 3, 4])
        0.5
        >>> ArrayJaccardSimilarity().score(["a", "b"], ["b", "a", "a"])
        1.0
        >>> report = evaluate({"tags": [1, 2, 3]}, {"tags": [2, 3, 4]},
        ...                   EvalConfig(metrics=[ArrayJaccardSimilarity()]))
        >>> float(report.metrics["array_jaccard_similarity"].representative())
        0.5
    """

    name = "array_jaccard_similarity"

    def compute(self, node: ArrayNode) -> float:
        """Set overlap between this node's two lists."""
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float:
        """`|A ∩ B| / |A ∪ B|` over the two values read as sets."""
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
