from __future__ import annotations

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats
from structured_eval.model.nodes.array_node import ArrayNode


class ArrayPRF1(ArrayMetric):
    """Array precision, recall and F1 in one pass.

    Returns a dict; the engine writes ``array_precision`` / ``array_recall`` /
    ``array_f1`` into ``report.metrics`` directly. Threshold and ``mode`` behave
    as for ``ArrayPrecision``.
    """

    name = "array_prf1"

    def __init__(self, threshold: float = 1.0, mode: stats.GradingMode = stats.GradingMode.HARD):
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)

    def compute(self, node: ArrayNode) -> dict[str, float]:
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return {
            "array_precision": p,
            "array_recall": r,
            "array_f1": stats.f1(p, r),
        }
