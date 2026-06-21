from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj
from structured_eval.model.nodes.object_node import ObjectNode


class ObjectRecall(ObjectMetric):
    """TP / (TP + FN) over an object's scalar fields.

    Match criterion and ``mode`` behave as for ``ObjectPrecision``.
    """

    name = "object_recall"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        threshold: Any = None,
        mode: stats.MatchMode = stats.MatchMode.HARD,
    ):
        self.score_policy = score_policy
        self.threshold = threshold
        self.mode = stats.MatchMode(mode)

    def compute(self, node: ObjectNode) -> float:
        verdicts = obj.matched_verdicts(node, self.score_policy, self.threshold)
        tp, predicted, expected = stats.prf_counts(
            verdicts, len(node.missing), len(node.spurious), self.mode
        )
        return stats.recall(tp, predicted, expected)
