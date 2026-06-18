from __future__ import annotations

from typing import Any

from structured_eval.alignment.base import ArrayAligner
from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.array_node import ArrayMatchResult


class ByIndexAligner(ArrayAligner):
    """Pairs the i-th expected item with the i-th actual item.

    For positionally significant lists (steps, time series, rankings). No key
    comparison is performed.
    """

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        n = min(len(expected), len(actual))
        return ArrayMatchResult(
            strategy=ArrayStrategy.BY_INDEX,
            matched=[(i, i) for i in range(n)],
            missed=list(range(n, len(expected))),
            spurious=list(range(n, len(actual))),
        )
