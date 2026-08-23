"""Positional array alignment — the `by_index` strategy."""

from __future__ import annotations

from typing import Any

from structured_eval.alignment.base import ArrayAligner
from structured_eval.models.config import ArrayStrategy
from structured_eval.models.nodes.array_node import ArrayMatchResult


class ByIndexAligner(ArrayAligner):
    """Pairs the i-th expected item with the i-th actual item.

    For positionally significant lists — steps, time series, rankings. No key
    is compared: the surplus of the longer side is simply unmatched.

    Example:
        >>> from structured_eval.alignment import ByIndexAligner
        >>> result = ByIndexAligner().align(["a", "b", "c"], ["a", "x"])
        >>> result.matched
        [(0, 0), (1, 1)]
        >>> result.missed, result.spurious
        ([2], [])
    """

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        """Pair equal positions, leaving the longer side's tail unmatched.

        Args:
            expected: The expected list.
            actual: The actual list.

        Returns:
            An `ArrayMatchResult` pairing `(i, i)` up to the shorter length.
        """
        n = min(len(expected), len(actual))
        return ArrayMatchResult(
            strategy=ArrayStrategy.BY_INDEX,
            matched=[(i, i) for i in range(n)],
            missed=list(range(n, len(expected))),
            spurious=list(range(n, len(actual))),
        )
