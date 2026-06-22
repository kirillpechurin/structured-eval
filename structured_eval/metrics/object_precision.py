from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj
from structured_eval.model.nodes.object_node import ObjectNode


class ObjectPrecision(ObjectMetric):
    """TP / (TP + FP) over an object's scalar fields.

    The match criterion per field is resolved from ``score_policy`` →
    field ``key_metric`` → ``ExactMatch`` (see ``_match_criterion``).
    ``mode="soft"`` drops the threshold and uses field scores fractionally.
    """

    name = "object_precision"

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

    def compute(self, node: ObjectNode) -> float:
        verdicts = obj.matched_verdicts(node, self.score_policy, self.threshold, self.weight_mode)
        tp, predicted, expected = stats.prf_counts(
            verdicts,
            obj.missing_weight(node, self.weight_mode),
            obj.spurious_weight(node, self.weight_mode),
            self.mode,
        )
        return stats.precision(tp, predicted, expected)
