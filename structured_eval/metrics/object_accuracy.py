"""The `object_accuracy` metric — weighted soft mean of an object's field scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj

if TYPE_CHECKING:
    from structured_eval.models.nodes.object_node import ObjectNode


class ObjectAccuracy(ObjectMetric):
    """Weighted soft mean of field correctness over an object's expected fields.

    Concepts:

    - TP (True Positive) — a `matched` entry, present on both sides;
    - FP (False Positive) — a `spurious` entry, produced but not expected;
    - FN (False Negative) — a `missing` entry, expected but not produced.

    Soft recall: `Σ weight·score / (matched_weight + missing_weight)`. Each
    matched field contributes its representative score, whatever its kind, or a
    `score_policy` override; missing expected fields count as 0.0.

    Spurious fields are **not** penalized — the denominator is the expected side
    only, so reach for `ObjectF1` when precision matters. An object expecting
    nothing is vacuously 1.0.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ObjectAccuracy
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": 2}, {"a": 1, "b": 9, "c": 3},
        ...                   EvalConfig(metrics=[ObjectAccuracy()]))
        >>> round(float(report.metrics["object_accuracy"].representative()), 3)
        0.333
    """

    name = "object_accuracy"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        weight_mode: stats.WeightMode = stats.WeightMode.PROPORTIONAL,
        name: str | None = None,
    ):
        """Configure the match criterion and how verdicts are counted.

        Args:
            score_policy: Per-field metric override, keyed by field name.
            weight_mode: How much each field counts:

                - `PROPORTIONAL` weighs it by its configured `weight`;
                - `NONE` gives every field 1.0.

            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.score_policy = score_policy
        self.weight_mode = stats.WeightMode(weight_mode)

    def compute(self, node: ObjectNode) -> float:
        """Weighted mean of this object's field scores."""
        verdicts = obj.matched_verdicts(
            node, self.score_policy, weight_mode=self.weight_mode
        )
        denom = sum(weight for _, _, weight in verdicts) + obj.missing_weight(
            node, self.weight_mode
        )
        if denom == 0:
            return 1.0
        return sum(weight * score for score, _, weight in verdicts) / denom
