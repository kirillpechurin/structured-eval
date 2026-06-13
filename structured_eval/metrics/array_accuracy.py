from __future__ import annotations

from structured_eval.metrics._array_stats import missing_spurious
from structured_eval.metrics._element_score import element_score
from structured_eval.metrics.protocol import ArrayMetric
from structured_eval.nodes.array_node import ArrayNode


class ArrayAccuracy(ArrayMetric):
    """Mean element score over the aligned items (soft).

    How good the matched elements are, regardless of how many were produced.
    Missed expected items count as 0.0; an empty/fully-missed array is
    vacuously 1.0.
    """

    name = "array_accuracy"

    def compute(self, node: ArrayNode) -> float:
        n_missing, _ = missing_spurious(node)
        denom = len(node.items) + n_missing
        if denom == 0:
            return 1.0
        return sum(element_score(item) for item in node.items) / denom
