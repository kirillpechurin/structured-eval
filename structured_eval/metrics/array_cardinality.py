"""The `array_cardinality` metric — agreement on how many items there are."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayCardinality(ArrayMetric):
    """Count agreement: `min(|actual|, |expected|) / max(...)`.

    A cheap length-ratio check independent of element correctness. Two empty
    arrays are vacuously 1.0.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayCardinality
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"xs": [1, 2]}, {"xs": [1, 9, 3]},
        ...                   EvalConfig(metrics=[ArrayCardinality()]))
        >>> round(float(report.metrics["array_cardinality"].representative()), 3)
        0.667
    """

    name = "array_cardinality"

    def compute(self, node: ArrayNode) -> float:
        """How close the two arrays are in length."""
        # Counted from the paired-up breakdown rather than from the raw lists:
        # a strategy matching by value can pair elements across positions, and
        # what this metric is about is how many there were on each side.
        actual_count = len(node.matched) + len(node.spurious)
        expected_count = len(node.matched) + len(node.missing)
        hi = max(actual_count, expected_count)
        return 1.0 if hi == 0 else min(actual_count, expected_count) / hi
