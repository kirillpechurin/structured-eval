from __future__ import annotations

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats
from structured_eval.model.nodes.array_node import ArrayNode


class ArrayPrecision(ArrayMetric):
    """TP / (TP + FP) over aligned array elements.

    An aligned item is a TP when its ``element_score`` clears ``threshold``
    (``mode="soft"`` instead adds the score fractionally); ``spurious`` items
    are FP. So a wrong-but-aligned element lowers precision.
    """

    name = "array_precision"

    def __init__(self, threshold: float = 1.0, mode: stats.GradingMode = stats.GradingMode.HARD):
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)

    def compute(self, node: ArrayNode) -> float:
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        return stats.precision(tp, predicted, expected)
