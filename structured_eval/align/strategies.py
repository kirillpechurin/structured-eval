from __future__ import annotations

from typing import Any

from structured_eval.core.config import ArrayStrategy
from structured_eval.metrics.field.exact import ExactMatch
from structured_eval.metrics.protocol import FieldMetric, get_metric_class
from structured_eval.nodes.array_node import ArrayMatchResult
from structured_eval.nodes.base import MISSING, _navigate

_MISSING_KEY = object()


def _resolve_metric(spec: Any) -> FieldMetric:
    if spec is None:
        return ExactMatch()
    return get_metric_class(spec)() if isinstance(spec, str) else spec


def _key_of(element: Any, key: str | None) -> Any:
    """The alignment key of an element: the whole element, or a named field."""
    if key is None:
        return element
    if isinstance(element, dict):
        value = _navigate(element, key)
        return None if value is MISSING else value
    return _MISSING_KEY


def align(
    expected: list,
    actual: list,
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX,
    key: str | None = None,
    key_metric: Any = None,
    key_threshold: float = 1.0,
) -> ArrayMatchResult:
    """Align an actual list against an expected list into matched/missed/spurious.

    ``BY_INDEX`` pairs the i-th with the i-th. ``BY_KEY`` (generalized) extracts
    a key from each element (``key`` field, or the whole element when ``key`` is
    None), compares keys with ``key_metric`` (default ``ExactMatch``) and pairs
    them greedily when the score clears ``key_threshold`` — this subsumes value-
    and similarity-based matching (technical_details_v3 §5).
    """
    if strategy == ArrayStrategy.BY_INDEX:
        return _by_index(expected, actual)
    return _by_key(expected, actual, key, _resolve_metric(key_metric), key_threshold)


def _by_index(expected: list, actual: list) -> ArrayMatchResult:
    n = min(len(expected), len(actual))
    return ArrayMatchResult(
        strategy=ArrayStrategy.BY_INDEX,
        matched=[(i, i) for i in range(n)],
        missed=list(range(n, len(expected))),
        spurious=list(range(n, len(actual))),
    )


def _by_key(
    expected: list,
    actual: list,
    key: str | None,
    metric: FieldMetric,
    threshold: float,
) -> ArrayMatchResult:
    used: set[int] = set()
    matched: list[tuple[int, int]] = []
    missed: list[int] = []

    for ei, e_item in enumerate(expected):
        e_key = _key_of(e_item, key)
        partner: int | None = None
        for ai, a_item in enumerate(actual):
            if ai in used:
                continue
            if metric.score(_key_of(a_item, key), e_key) >= threshold:
                partner = ai
                break
        if partner is None:
            missed.append(ei)
        else:
            used.add(partner)
            matched.append((ei, partner))

    spurious = [ai for ai in range(len(actual)) if ai not in used]
    return ArrayMatchResult(
        strategy=ArrayStrategy.BY_KEY, matched=matched, missed=missed, spurious=spurious
    )
