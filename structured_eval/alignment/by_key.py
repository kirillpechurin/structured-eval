from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.alignment.base import ArrayAligner, key_value, normalize_key
from structured_eval.metrics.base import BaseMetric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.models.config import ArrayStrategy
from structured_eval.models.nodes.array_node import ArrayMatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class ByKeyAligner(ArrayAligner):
    """Pairs items whose keys match, greedily best-first (generalized matching).

    Extracts a key from each element (the ``key`` field, or the whole element
    when ``key`` is None), compares keys with ``key_metric`` (default
    ``ExactMatch``) and pairs them when the score clears ``threshold``. This
    subsumes value- and similarity-based matching (technical_details_v3 §5).

    ``key`` may also name **several fields** — a composite key such as
    ``["sku", "warehouse"]``. Each field is scored with ``key_metric`` and the
    element's key score is the mean over the fields, so with the default
    ``ExactMatch`` and ``threshold=1.0`` every field must match; a soft
    ``key_metric`` lets a strong field carry a weaker one. A one-field key is
    the mean of one score, i.e. identical to passing that field as a string.

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
        key: str | Sequence[str] | None = None,
        key_metric: str | BaseMetric | None = None,
        threshold: float = 1.0,
    ):
        self.key = normalize_key(key, self.__class__.__name__)
        metric = ExactMatch() if key_metric is None else resolve_metric(key_metric)
        self.scorer = MetricInvoker(metric)
        self.threshold = threshold

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        # Score every (expected, actual) pair on its key; keep those clearing
        # the threshold. Generated in (ei, ai) order so a stable sort breaks
        # score ties by that order (→ exact-key matches reproduce first-match).
        e_keys = [self._key_of(item) for item in expected]
        a_keys = [self._key_of(item) for item in actual]
        candidates: list[tuple[float, int, int]] = []
        for ei, e_key in enumerate(e_keys):
            for ai, a_key in enumerate(a_keys):
                score = self._key_score(e_key, a_key)
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

    # ── key extraction & scoring ────────────────────────────────────────────

    def _key_of(self, element: Any) -> list[Any]:
        """The element's key: one value per configured field, or the element."""
        if self.key is None:
            return [element]
        return [key_value(element, field) for field in self.key]

    def _key_score(self, e_key: list[Any], a_key: list[Any]) -> float:
        """Mean of the per-field key scores (a one-field key is that score)."""
        total = sum(
            self.scorer.scalar_on_values(a, e)
            for e, a in zip(e_key, a_key, strict=True)
        )
        return total / len(e_key)
