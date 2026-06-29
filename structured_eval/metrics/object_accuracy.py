from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj

if TYPE_CHECKING:
    from structured_eval.model.nodes.object_node import ObjectNode


class ObjectAccuracy(ObjectMetric):
    """Weighted soft mean of field correctness over an object's expected fields.

    Equivalent to **soft recall**: ``Σ weight·score / (matched_weight +
    missing_weight)``. Each matched field contributes its ``representative``
    (any child kind — a nested object/array counts via its representative, not
    only scalars), or a ``score_policy`` override. Missing expected fields count
    as 0.0. **Spurious (extra) fields are not penalized** — the denominator is
    the expected side only (use ``ObjectF1`` for a precision-aware score). An
    object with no expected fields is vacuously 1.0.

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
        verdicts = obj.matched_verdicts(
            node, self.score_policy, weight_mode=self.weight_mode
        )
        denom = sum(weight for _, _, weight in verdicts) + obj.missing_weight(
            node, self.weight_mode
        )
        if denom == 0:
            return 1.0
        return sum(weight * score for score, _, weight in verdicts) / denom
