from __future__ import annotations

from typing import Any

from structured_eval.alignment.base import ArrayAligner, key_value
from structured_eval.metrics.base import Metric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.array_node import ArrayMatchResult


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
        self.metric: Metric[Any] = (
            ExactMatch() if key_metric is None else resolve_metric(key_metric)
        )
        self.threshold = threshold

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        used: set[int] = set()
        matched: list[tuple[int, int]] = []
        missed: list[int] = []

        for ei, e_item in enumerate(expected):
            e_key = key_value(e_item, self.key)
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
            key_score = self.metric.score(key_value(a_item, self.key), e_key)
            assert not isinstance(key_score, dict)  # criterion metrics return a scalar
            if key_score >= self.threshold:
                return ai
        return None
