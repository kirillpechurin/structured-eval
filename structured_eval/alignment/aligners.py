from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_eval.metrics.base import FieldMetric, get_metric_class
from structured_eval.metrics.exact import ExactMatch
from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.array_node import ArrayMatchResult
from structured_eval.model.nodes.base import MISSING, _navigate

_MISSING_KEY = object()


def _resolve_metric(spec: Any) -> FieldMetric:
    if spec is None:
        return ExactMatch()
    if isinstance(spec, str):
        instance = get_metric_class(spec)()
        assert isinstance(instance, FieldMetric)
        return instance
    assert isinstance(spec, FieldMetric)
    return spec


class ArrayAligner(ABC):
    """Maps actual array items onto expected ones (the only role of a matcher).

    ``align`` returns an ``ArrayMatchResult`` with matched ``(expected_idx,
    actual_idx)`` pairs plus the unmatched expected (missed) and actual
    (spurious) indices. Value scoring of matched pairs happens later, in the
    array metrics.
    """

    @abstractmethod
    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult: ...


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


class ByKeyAligner(ArrayAligner):
    """Greedily pairs items whose keys match (generalized matching).

    Extracts a key from each element (the ``key`` field, or the whole element
    when ``key`` is None), compares keys with ``key_metric`` (default
    ``ExactMatch``) and pairs them when the score clears ``threshold``. This
    subsumes value- and similarity-based matching (technical_details_v3 §5).
    """

    def __init__(
        self,
        key: str | None = None,
        key_metric: Any = None,
        threshold: float = 1.0,
    ):
        self.key = key
        self.metric = _resolve_metric(key_metric)
        self.threshold = threshold

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        used: set[int] = set()
        matched: list[tuple[int, int]] = []
        missed: list[int] = []

        for ei, e_item in enumerate(expected):
            e_key = self._key_of(e_item)
            partner = self._find_partner(actual, e_key, used)
            if partner is None:
                missed.append(ei)
            else:
                used.add(partner)
                matched.append((ei, partner))

        spurious = [ai for ai in range(len(actual)) if ai not in used]
        return ArrayMatchResult(
            strategy=ArrayStrategy.BY_KEY,
            matched=matched,
            missed=missed,
            spurious=spurious,
        )

    def _find_partner(self, actual: list[Any], e_key: Any, used: set[int]) -> int | None:
        for ai, a_item in enumerate(actual):
            if ai in used:
                continue
            key_score = self.metric.score(self._key_of(a_item), e_key)
            assert not isinstance(key_score, dict)  # criterion metrics return a scalar
            if key_score >= self.threshold:
                return ai
        return None

    def _key_of(self, element: Any) -> Any:
        """The alignment key of an element: the whole element, or a named field."""
        if self.key is None:
            return element
        if isinstance(element, dict):
            value = _navigate(element, self.key)
            return None if value is MISSING else value
        return _MISSING_KEY


def make_aligner(
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX,
    key: str | None = None,
    key_metric: Any = None,
    key_threshold: float = 1.0,
) -> ArrayAligner:
    """Build the aligner for an array config's strategy."""
    if strategy == ArrayStrategy.BY_INDEX:
        return ByIndexAligner()
    return ByKeyAligner(key, key_metric, key_threshold)
