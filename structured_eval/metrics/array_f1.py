"""The `array_f1` metric — harmonic mean of array precision and recall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayF1(ArrayMetric):
    """Harmonic mean of array precision and recall over aligned elements.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayF1
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayF1()]))
        >>> round(float(report.metrics["array_f1"].representative()), 3)
        0.4
    """

    name = "array_f1"

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
        """Harmonic mean of this array's element precision and recall."""
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return stats.f1(p, r)
