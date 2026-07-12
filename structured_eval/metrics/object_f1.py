from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj

if TYPE_CHECKING:
    from structured_eval.models.nodes.object_node import ObjectNode


class ObjectF1(ObjectMetric):
    """Harmonic mean of object precision and recall over an object's fields.

    Slot-filling F1: matched-and-correct → TP, missing → FN, extra → FP. Match
    criterion and ``mode`` behave as for ``ObjectPrecision`` (counts all child
    kinds via their representative).
    """

    name = "object_f1"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        threshold: float | None = None,
        mode: stats.GradingMode = stats.GradingMode.HARD,
        weight_mode: stats.WeightMode = stats.WeightMode.PROPORTIONAL,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.score_policy = score_policy
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)
        self.weight_mode = stats.WeightMode(weight_mode)

    def compute(self, node: ObjectNode) -> float:
        verdicts = obj.matched_verdicts(
            node, self.score_policy, self.threshold, self.weight_mode
        )
        tp, predicted, expected = stats.prf_counts(
            verdicts,
            obj.missing_weight(node, self.weight_mode),
            obj.spurious_weight(node, self.weight_mode),
            self.mode,
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return stats.f1(p, r)
