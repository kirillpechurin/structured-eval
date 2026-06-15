"""Verdicts and counts for array metrics, reusing the object P/R/F1 arithmetic.

An aligned item is graded by its recursive ``element_score`` against a single
``threshold`` (hard) or contributes that score fractionally (soft) — mirroring
how object fields are graded. ``missed`` items are FN, ``spurious`` items FP.
"""

from __future__ import annotations

from structured_eval.metrics._shared.element_score import element_score
from structured_eval.model.nodes.array_node import ArrayNode


def verdicts(node: ArrayNode, threshold: float) -> list[tuple[float, float]]:
    return [(element_score(item), threshold) for item in node.items]


def missing_spurious(node: ArrayNode) -> tuple[int, int]:
    mr = node.match_result
    if mr is None:
        return 0, 0
    return len(mr.missed), len(mr.spurious)
