from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj
from structured_eval.model.nodes.object_node import ObjectNode


class ObjectAccuracy(ObjectMetric):
    """Mean field score over an object's expected scalar fields (soft).

    Continuous (rewards partial matches), unlike threshold-based F1. The per
    field score uses the resolved match criterion (``score_policy`` → field
    ``key_metric`` → ``ExactMatch``). Missing expected fields count as 0.0; an
    object with no expected scalar fields is vacuously 1.0.

    ``weight_mode`` (default ``PROPORTIONAL``) makes this a weighted mean by each
    child's configured ``weight``; ``NONE`` restores the plain mean.
    """

    name = "object_accuracy"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        weight_mode: stats.WeightMode = stats.WeightMode.PROPORTIONAL,
    ):
        self.score_policy = score_policy
        self.weight_mode = stats.WeightMode(weight_mode)

    def compute(self, node: ObjectNode) -> float:
        verdicts = obj.matched_verdicts(node, self.score_policy, weight_mode=self.weight_mode)
        denom = sum(weight for _, _, weight in verdicts) + obj.missing_weight(
            node, self.weight_mode
        )
        if denom == 0:
            return 1.0
        return sum(weight * score for score, _, weight in verdicts) / denom
