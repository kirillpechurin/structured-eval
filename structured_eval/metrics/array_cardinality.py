"""The `array_cardinality` metric — agreement on how many items there are."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


class ArrayCardinality(ArrayMetric):
    """Count agreement: ``min(|actual|, |expected|) / max(...)``.

    A cheap length-ratio check independent of element correctness. Two empty
    arrays are vacuously 1.0.
    """

    name = "array_cardinality"

    def compute(self, node: ArrayNode) -> float:
        # Counted from the paired-up breakdown rather than from the raw lists:
        # a strategy matching by value can pair elements across positions, and
        # what this metric is about is how many there were on each side.
        actual_count = len(node.matched) + len(node.spurious)
        expected_count = len(node.matched) + len(node.missing)
        hi = max(actual_count, expected_count)
        return 1.0 if hi == 0 else min(actual_count, expected_count) / hi
