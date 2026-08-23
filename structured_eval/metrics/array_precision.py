"""The `array_precision` metric — TP / (TP + FP) over aligned array elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayPrecision(ArrayMetric):
    """TP / (TP + FP) over aligned array elements.

    Concepts:

    - TP (True Positive) — a `matched` entry, present on both sides;
    - FP (False Positive) — a `spurious` entry, produced but not expected;
    - FN (False Negative) — a `missing` entry, expected but not produced.

    An aligned item is a TP when its `score` clears `threshold`
    (`mode="soft"` instead adds the score fractionally);

    `spurious` items are FP. So a wrong-but-aligned element lowers precision.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayPrecision
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayPrecision()]))
        >>> round(float(report.metrics["array_precision"].representative()), 3)
        0.5
    """

    name = "array_precision"

    def __init__(
        self,
        threshold: float = 1.0,
        mode: stats.GradingMode = stats.GradingMode.HARD,
        name: str | None = None,
    ):
        """Set the bar each aligned element must clear.

        Args:
            threshold: The score an aligned element needs to count as a match.
            mode: How an element counts toward TP:

                - `HARD` counts it only once it clears the threshold;
                - `SOFT` counts its score fractionally, ignoring the threshold.

            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.mode = stats.GradingMode(mode)

    def compute(self, node: ArrayNode) -> float:
        """TP / (TP + FP) over this array's aligned elements."""
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        return stats.precision(tp, predicted, expected)
