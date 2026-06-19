from __future__ import annotations

from typing import Any

from structured_eval.metrics._shared import match_criterion as mc
from structured_eval.metrics._shared import object_stats as stats
from structured_eval.metrics.base import ObjectMetric
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
        mode: str = "hard",
    ):
        self.score_policy = score_policy
        self.threshold = threshold
        self.mode = mode

    def compute(self, node: ObjectNode) -> float:
        verdicts = mc.matched_verdicts(node, self.score_policy, self.threshold)
        tp, predicted, expected = stats.prf_counts(
            verdicts, len(node.missing), len(node.spurious), self.mode
        )
        return stats.precision(tp, predicted, expected)
