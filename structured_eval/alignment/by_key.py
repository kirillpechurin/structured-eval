"""Key-based array alignment — the `by_key` strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.alignment.base import (
    ArrayAligner,
    key_value,
    keyable,
    normalize_key,
)
from structured_eval.metrics.base import BaseMetric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.models.config import ArrayStrategy
from structured_eval.models.nodes.array_node import ArrayMatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class ByKeyAligner(ArrayAligner):
    """Pairs items whose keys match, greedily best-first (generalized matching).

    Extracts a key from each element — the `key` field, or the whole element
    when `key` is `None` — compares keys with `key_metric` and pairs them when
    the score clears `threshold`. Matching by value and matching by similarity
    are both this strategy, differing only in the metric.

    A composite `key` such as `["sku", "warehouse"]` scores each field with
    `key_metric` and takes their mean, so with the default `ExactMatch` and
    `threshold=1.0` every field must match, while a soft `key_metric` lets a
    strong field carry a weaker one.

    Pairing is **globally greedy**: every candidate pair clearing the threshold
    is ranked by score, highest first, and claimed one-to-one. A soft key
    therefore picks the strongest available partner rather than the first one
    found, and the outcome does not depend on element order.

    With an exact key every passing score ties at 1.0, and this reduces to
    first-match. It is the cheap, scipy-free approximation of the optimal
    assignment `HungarianAligner` computes.

    Example:
        >>> from structured_eval.alignment import ByKeyAligner
        >>> expected = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 5}]
        >>> actual = [{"sku": "B-2", "qty": 5}, {"sku": "C-3", "qty": 1}]
        >>> result = ByKeyAligner(key="sku").align(expected, actual)
        >>> result.matched          # B-2 pairs across the reordering
        [(1, 0)]
        >>> result.missed, result.spurious
        ([0], [1])
    """

    def __init__(
        self,
        key: str | Sequence[str] | None = None,
        key_metric: str | BaseMetric | None = None,
        threshold: float = 1.0,
    ):
        """Set what the key is, how it is compared, and how close counts.

        Args:
            key: Field path to key on, several of them for a composite key, or
                `None` to key on the whole element.
            key_metric: Metric comparing two keys, by instance or registered
                name. Defaults to `ExactMatch`.
            threshold: Key score at which a pair may be claimed.

        Raises:
            ValueError: If `key` is a sequence that names no field at all.
        """
        self.key = normalize_key(key, self.__class__.__name__)
        metric = ExactMatch() if key_metric is None else resolve_metric(key_metric)
        self.scorer = MetricInvoker(metric)
        self.threshold = threshold

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        """Claim the best-scoring key pairs one-to-one, best first.

        Args:
            expected: The expected list.
            actual: The actual list.

        Returns:
            An `ArrayMatchResult` whose pairs are reported in expected order.
        """
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
        if not (keyable(e_key) and keyable(a_key)):
            return 0.0
        total = sum(
            self.scorer.scalar_on_values(a, e)
            for e, a in zip(e_key, a_key, strict=True)
        )
        return total / len(e_key)
