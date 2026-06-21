from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj
from structured_eval.model.nodes.object_node import ObjectNode


class ObjectF1(ObjectMetric):
    """Harmonic mean of object precision and recall over scalar fields.

    Match criterion and ``mode`` behave as for ``ObjectPrecision``.
    """

    name = "object_f1"

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
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return stats.f1(p, r)
