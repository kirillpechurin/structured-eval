"""The `array_prf1` metric — array precision, recall and F1 from one pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric
from structured_eval.metrics.utils import array as astats
from structured_eval.metrics.utils import calculate as stats

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayPRF1(ArrayMetric):
    """Array precision, recall and F1 in one pass.

    Returns a dict; the engine writes:

    - `array_precision`;
    - `array_recall`;
    - `array_f1`;

    into `report.metrics` directly.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayPRF1
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayPRF1()]))
        >>> round(float(report.metrics["array_f1"].representative()), 3)
        0.4
    """

    name = "array_prf1"

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

    def compute(self, node: ArrayNode) -> dict[str, float]:
        """Precision, recall and F1 for this array, in one dict."""
        n_missing, n_spurious = astats.missing_spurious(node)
        tp, predicted, expected = stats.prf_counts(
            astats.verdicts(node, self.threshold), n_missing, n_spurious, self.mode
        )
        p = stats.precision(tp, predicted, expected)
        r = stats.recall(tp, predicted, expected)
        return {
            "array_precision": p,
            "array_recall": r,
            "array_f1": stats.f1(p, r),
        }
