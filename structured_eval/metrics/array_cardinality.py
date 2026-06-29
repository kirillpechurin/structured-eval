from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ArrayMetric

if TYPE_CHECKING:
    from structured_eval.model.nodes.array_node import ArrayNode


class ArrayCardinality(ArrayMetric):
    """Count agreement: ``min(|actual|, |expected|) / max(...)``.

    A cheap length-ratio check independent of element correctness. Two empty
    arrays are vacuously 1.0.
    """

    name = "array_cardinality"

    def compute(self, node: ArrayNode) -> float:
        mr = node.match_result
        if mr is None:
            return 1.0
        actual_count = len(mr.matched) + len(mr.spurious)
        expected_count = len(mr.matched) + len(mr.missed)
        hi = max(actual_count, expected_count)
        return 1.0 if hi == 0 else min(actual_count, expected_count) / hi
