from __future__ import annotations

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats
from structured_eval.model.nodes.array_node import ArrayNode


class ArrayRecall(ArrayMetric):
    """TP / (TP + FN) over aligned array elements.

    ``missed`` expected items are FN; threshold and ``mode`` behave as for
    ``ArrayPrecision``.
    """

    name = "array_recall"

    def __init__(self, threshold: float = 1.0, mode: stats.GradingMode = stats.GradingMode.HARD):
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)

    def compute(self, node: ArrayNode) -> float:
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        return stats.recall(tp, predicted, expected)
