"""Verdicts for array metrics: aligned items → ``(score, threshold)`` pairs.

An aligned item is graded by its representative score against a single
``threshold`` (hard) or contributes that score fractionally (soft) — mirroring
how object fields are graded. ``missed`` items are FN, ``spurious`` items FP.
The verdicts feed ``calculate.prf_counts``.
"""

from __future__ import annotations

from structured_eval.model.nodes.array_node import ArrayNode


def verdicts(node: ArrayNode, threshold: float) -> list[tuple[float, float]]:
    """``(representative, threshold)`` for each aligned item of an array."""
    return [(item.representative, threshold) for item in node.items]


def missing_spurious(node: ArrayNode) -> tuple[int, int]:
    """``(n_missed, n_spurious)`` from the array's alignment result."""
    mr = node.match_result
    if mr is None:
        return 0, 0
    return len(mr.missed), len(mr.spurious)
