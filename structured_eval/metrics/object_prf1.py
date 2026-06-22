from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj
from structured_eval.model.nodes.object_node import ObjectNode


class ObjectPRF1(ObjectMetric):
    """Precision, recall and F1 in one pass.

    Returns a dict; the engine writes each key (``object_precision``,
    ``object_recall``, ``object_f1``) into ``report.metrics`` directly. Match
    criterion and ``mode`` behave as for ``ObjectPrecision``.
    """

    name = "object_prf1"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        threshold: Any = None,
        mode: stats.GradingMode = stats.GradingMode.HARD,
        weight_mode: stats.WeightMode = stats.WeightMode.PROPORTIONAL,
    ):
        self.score_policy = score_policy
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)
        self.weight_mode = stats.WeightMode(weight_mode)

    def compute(self, node: ObjectNode) -> dict[str, float]:
        verdicts = obj.matched_verdicts(node, self.score_policy, self.threshold, self.weight_mode)
        tp, predicted, expected = stats.prf_counts(
            verdicts,
            obj.missing_weight(node, self.weight_mode),
            obj.spurious_weight(node, self.weight_mode),
            self.mode,
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return {
            "object_precision": p,
            "object_recall": r,
            "object_f1": stats.f1(p, r),
        }
