from __future__ import annotations

from collections import Counter
from typing import Any

# ── Primitive array metrics ───────────────────────────────────────────────────
#
# All functions operate on flat lists of hashable values.
# Object arrays (list[dict]) are handled by field_accuracy._compare_array_by_index.


def array_exact_match(actual: list[Any], expected: list[Any]) -> float:
    """1.0 iff actual equals expected element-by-element (order and length matter)."""
    return 1.0 if actual == expected else 0.0


def array_set_match(actual: list[Any], expected: list[Any]) -> float:
    """F1 over multisets: order-independent, duplicates counted.

    Uses Counter-based intersection so that repeated elements are handled
    correctly (e.g. [1, 1, 2] ∩ [1, 2, 2] = [1, 2], not [1, 1, 2]).
    Returns 1.0 when both lists are empty.
    """
    if not actual and not expected:
        return 1.0

    recall = array_element_recall(actual, expected)
    precision = array_element_precision(actual, expected)
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def array_element_recall(actual: list[Any], expected: list[Any]) -> float:
    """Fraction of expected elements found in actual (multiset recall).

    Returns 1.0 when expected is empty (nothing to recall).
    """
    if not expected:
        return 1.0
    intersection = _multiset_intersection(actual, expected)
    return intersection / len(expected)


def array_element_precision(actual: list[Any], expected: list[Any]) -> float:
    """Fraction of actual elements that appear in expected (multiset precision).

    Returns 1.0 when actual is empty (nothing to be imprecise about).
    """
    if not actual:
        return 1.0
    intersection = _multiset_intersection(actual, expected)
    return intersection / len(actual)


# ── Internals ─────────────────────────────────────────────────────────────────


def _multiset_intersection(a: list[Any], b: list[Any]) -> int:
    """Sum of min(count_a[x], count_b[x]) over all unique elements."""
    ca = Counter(a)
    cb = Counter(b)
    return sum((ca & cb).values())
