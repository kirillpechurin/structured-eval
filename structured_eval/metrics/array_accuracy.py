"""The `array_accuracy` metric — mean element score over aligned array items."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayAccuracy(ArrayMetric):
    """Mean element score over the aligned items (soft).

    How good the matched elements are, regardless of how many were produced:
    the mean of each matched item's score over `matched` plus `missing`. The
    default array metric.

    Grades `node.matched`, not `node.items` — an element with no expected
    counterpart has nothing to be accurate *against*, and penalizing it is
    `ArrayF1`'s job.

    Nothing matched means nothing was expected, so there is nothing to be
    inaccurate about and the array scores 1.0. Spurious elements do not pull it
    down at all — that is `ArrayF1`'s job, not this metric's.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayAccuracy
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayAccuracy()]))
        >>> round(float(report.metrics["array_accuracy"].representative()), 3)
        0.333
    """

    name = "array_accuracy"

    def compute(self, node: ArrayNode) -> float:
        """Mean score over this array's aligned elements."""
        denom = len(node.matched) + len(node.missing)
        if denom == 0:
            return 1.0
        return sum(item.representative for item in node.matched) / denom
