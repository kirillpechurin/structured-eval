"""The `object_recall` metric — TP / (TP + FN) over an object's fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.utils import calculate as stats
from structured_eval.metrics.utils import object_utils as obj

if TYPE_CHECKING:
    from structured_eval.models.nodes.object_node import ObjectNode


class ObjectRecall(ObjectMetric):
    """TP / (TP + FN) over an object's fields (slot-filling recall).

    Concepts:

    - TP (True Positive) — a `matched` entry, present on both sides;
    - FP (False Positive) — a `spurious` entry, produced but not expected;
    - FN (False Negative) — a `missing` entry, expected but not produced.

    Missing expected fields are FN; extra ones are not counted here.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ObjectRecall
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": 2}, {"a": 1, "b": 9, "c": 3},
        ...                   EvalConfig(metrics=[ObjectRecall()]))
        >>> round(float(report.metrics["object_recall"].representative()), 3)
        0.333
    """

    name = "object_recall"

    def __init__(
        self,
        score_policy: dict[str, Any] | None = None,
        threshold: float | dict[str, float] | None = None,
        mode: stats.GradingMode = stats.GradingMode.HARD,
        weight_mode: stats.WeightMode = stats.WeightMode.PROPORTIONAL,
        name: str | None = None,
    ):
        """Configure the match criterion and how verdicts are counted.

        Args:
            score_policy: Per-field metric override, keyed by field name.
            threshold: The bar a field must clear; one float, or a per-field dict.
            mode: How a field counts toward TP:

                - `HARD` counts it only once it clears its threshold;
                - `SOFT` counts its score fractionally, ignoring the threshold.

            weight_mode: How much each field counts:

                - `PROPORTIONAL` weighs it by its configured `weight`;
                - `NONE` gives every field 1.0.

            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.score_policy = score_policy
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)
        self.weight_mode = stats.WeightMode(weight_mode)

    def compute(self, node: ObjectNode) -> float:
        """TP / (TP + FN) over this object's fields."""
        verdicts = obj.matched_verdicts(
            node, self.score_policy, self.threshold, self.weight_mode
        )
        tp, predicted, expected = stats.prf_counts(
            verdicts,
            obj.missing_weight(node, self.weight_mode),
            obj.spurious_weight(node, self.weight_mode),
            self.mode,
        )
        return stats.recall(tp, predicted, expected)
