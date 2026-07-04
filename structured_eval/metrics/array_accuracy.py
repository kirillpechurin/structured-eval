from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayAccuracy(ArrayMetric):
    """Mean element score over the aligned items (soft).

    How good the matched elements are, regardless of how many were produced:
    the mean of each matched item's representative score over (items + missed).
    Missed expected items count as 0.0; an empty/fully-missed array is vacuously
    1.0. The default array metric, and the array branch of the old
    ``structural_score``.
    """

    name = "array_accuracy"

    def compute(self, node: ArrayNode) -> float:
        n_missing = len(node.match_result.missed) if node.match_result else 0
        denom = len(node.items) + n_missing
        if denom == 0:
            return 1.0
        return sum(item.representative for item in node.items) / denom
