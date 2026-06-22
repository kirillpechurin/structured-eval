from __future__ import annotations

from typing import Any

from structured_eval.alignment.base import ArrayAligner, key_value
from structured_eval.metrics.base import resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.array_node import ArrayMatchResult


class ByKeyAligner(ArrayAligner):
    """Pairs items whose keys match, greedily best-first (generalized matching).

    Extracts a key from each element (the ``key`` field, or the whole element
    when ``key`` is None), compares keys with ``key_metric`` (default
    ``ExactMatch``) and pairs them when the score clears ``threshold``. This
    subsumes value- and similarity-based matching (technical_details_v3 §5).

    Pairing is **globally greedy**: every candidate pair whose key score clears
    the threshold is ranked by score (highest first) and claimed one-to-one,
    skipping pairs whose either side is already taken. So a *soft* key picks the
    strongest available partner rather than the first one found, and the result
    does not depend on element order. With an exact key (all passing scores tie
    at 1.0) this reduces to the original first-match behaviour. It is a cheap,
    scipy-free approximation of the optimal assignment that ``HungarianAligner``
    computes.
    """

    def __init__(
        self,
        key: str | None = None,
        key_metric: Any = None,
        threshold: float = 1.0,
    ):
        self.key = key
        metric = ExactMatch() if key_metric is None else resolve_metric(key_metric)
        self.scorer = MetricInvoker(metric)
        self.threshold = threshold

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        # Score every (expected, actual) pair on its key; keep those clearing
        # the threshold. Generated in (ei, ai) order so a stable sort breaks
        # score ties by that order (→ exact-key matches reproduce first-match).
        candidates: list[tuple[float, int, int]] = []
        for ei, e_item in enumerate(expected):
            e_key = key_value(e_item, self.key)
            for ai, a_item in enumerate(actual):
                score = self.scorer.scalar_on_values(key_value(a_item, self.key), e_key)
                if score >= self.threshold:
                    candidates.append((score, ei, ai))
        candidates.sort(key=lambda c: c[0], reverse=True)  # best first; ties keep order

        used_e: set[int] = set()
        used_a: set[int] = set()
        matched: list[tuple[int, int]] = []
        for _score, ei, ai in candidates:
            if ei in used_e or ai in used_a:
                continue
            used_e.add(ei)
            used_a.add(ai)
            matched.append((ei, ai))
        matched.sort()  # report pairs in expected order

        missed = [ei for ei in range(len(expected)) if ei not in used_e]
        spurious = [ai for ai in range(len(actual)) if ai not in used_a]
        return ArrayMatchResult(
            strategy=ArrayStrategy.BY_KEY,
            matched=matched,
            missed=missed,
            spurious=spurious,
        )
