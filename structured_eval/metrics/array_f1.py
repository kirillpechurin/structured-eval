from __future__ import annotations

from structured_eval.metrics._shared import array_stats as astats
from structured_eval.metrics._shared import object_stats as stats
from structured_eval.metrics.base import ArrayMetric
from structured_eval.model.nodes.array_node import ArrayNode


class ArrayF1(ArrayMetric):
    """Harmonic mean of array precision and recall over aligned elements.

    Threshold and ``mode`` behave as for ``ArrayPrecision``.
    """

    name = "array_f1"

    def __init__(self, threshold: float = 1.0, mode: str = "hard"):
        self.threshold = threshold
        self.mode = mode

    def compute(self, node: ArrayNode) -> float:
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return stats.f1(p, r)
