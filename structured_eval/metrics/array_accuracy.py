from __future__ import annotations

from structured_eval.metrics._shared.match_criterion import structural_score
from structured_eval.metrics.base import ArrayMetric
from structured_eval.model.nodes.array_node import ArrayNode


class ArrayAccuracy(ArrayMetric):
    """Mean element score over the aligned items (soft).

    How good the matched elements are, regardless of how many were produced.
    Missed expected items count as 0.0; an empty/fully-missed array is
    vacuously 1.0. This is exactly the array branch of ``structural_score``.
    """

    name = "array_accuracy"

    def compute(self, node: ArrayNode) -> float:
        return structural_score(node)
