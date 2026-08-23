"""The `array_recall` metric — TP / (TP + FN) over aligned array elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayRecall(ArrayMetric):
    """TP / (TP + FN) over aligned array elements.

    Concepts:

    - TP (True Positive) — a `matched` entry, present on both sides;
    - FP (False Positive) — a `spurious` entry, produced but not expected;
    - FN (False Negative) — a `missing` entry, expected but not produced.

    Expected items with no counterpart are FN; spurious ones are not counted.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayRecall
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayRecall()]))
        >>> round(float(report.metrics["array_recall"].representative()), 3)
        0.333
    """

    name = "array_recall"

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
        """TP / (TP + FN) over this array's aligned elements."""
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        return stats.recall(tp, predicted, expected)
