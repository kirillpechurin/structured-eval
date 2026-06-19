from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from structured_eval.alignment.base import ArrayAligner, key_value
from structured_eval.metrics.base import FieldMetric, Metric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.numeric_closeness import NumericCloseness
from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.array_node import ArrayMatchResult

try:
    import rapidfuzz  # noqa: F401

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

_LARGE_MATRIX_WARN = 10_000  # rows*cols beyond which we warn (quadratic scoring cost)

# A per-element similarity: a Metric instance (every Metric has ``score``), its
# registered name, or a plain ``(actual, expected) -> float`` callable.
Scorer = Metric[Any] | str | Callable[[Any, Any], float]


class HungarianAligner(ArrayAligner):
    """Optimal one-to-one assignment via the Hungarian algorithm.

    Builds a similarity matrix ``S[i,j] = score(expected[i], actual[j])`` and
    solves ``min sum(1 - S)`` with ``scipy.optimize.linear_sum_assignment`` —
    the globally optimal pairing regardless of order. A pair counts as matched
    only when its similarity clears ``threshold`` (otherwise both sides are left
    unmatched: a missed expected and a spurious actual).

    ``scorer`` is the element similarity. Crucially our field metrics already
    *are* scorers (``FieldMetric.score(actual, expected) -> float``), so no
    adapter is needed — a metric, its registered name, or a plain callable is
    used directly. It may be:

    * a single ``Scorer`` — applied to the whole element;
    * a ``dict[str, Scorer]`` — per-field scorers for arrays of objects; the
      element score is the mean over the union of fields (a field with no entry
      falls back to its type default);
    * ``None`` — type-aware default (graded numeric / ``Fuzzy`` / exact), with
      objects scored field-by-field.

    ``key`` scores on a named sub-field instead of the whole element. Requires
    the ``align`` extra (scipy).
    """

    def __init__(
        self,
        scorer: Scorer | dict[str, Scorer] | None = None,
        threshold: float = 0.8,
        key: str | None = None,
    ):
        self.scorer = scorer
        self.threshold = threshold
        self.key = key

    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        if not expected or not actual:
            return ArrayMatchResult(
                strategy=ArrayStrategy.HUNGARIAN,
                matched=[],
                missed=list(range(len(expected))),
                spurious=list(range(len(actual))),
            )

        if len(expected) * len(actual) > _LARGE_MATRIX_WARN:
            warnings.warn(
                f"HungarianAligner: large {len(expected)}x{len(actual)} similarity "
                "matrix; alignment may be slow.",
                stacklevel=2,
            )

        from scipy.optimize import linear_sum_assignment

        cost = [[1.0 - self._score(e, a) for a in actual] for e in expected]
        rows, cols = linear_sum_assignment(cost)

        matched: list[tuple[int, int]] = []
        used_e: set[int] = set()
        used_a: set[int] = set()
        for ei, ai in zip(rows, cols, strict=True):
            if 1.0 - cost[ei][ai] >= self.threshold:
                matched.append((int(ei), int(ai)))
                used_e.add(int(ei))
                used_a.add(int(ai))

        missed = [ei for ei in range(len(expected)) if ei not in used_e]
        spurious = [ai for ai in range(len(actual)) if ai not in used_a]
        return ArrayMatchResult(
            strategy=ArrayStrategy.HUNGARIAN,
            matched=matched,
            missed=missed,
            spurious=spurious,
        )

    # ── element similarity ──────────────────────────────────────────────────

    def _score(self, expected: Any, actual: Any) -> float:
        return self._similarity(
            key_value(expected, self.key),
            key_value(actual, self.key),
            self.scorer,
        )

    def _similarity(
        self,
        expected: Any,
        actual: Any,
        scorer: Scorer | dict[str, Scorer] | None,
    ) -> float:
        if isinstance(scorer, dict):
            return self._object_similarity(expected, actual, scorer)
        if scorer is not None:
            return self._apply(scorer, expected, actual)
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._object_similarity(expected, actual, {})
        return self._apply(self._default_scorer(expected, actual), expected, actual)

    def _object_similarity(self, expected: Any, actual: Any, scorers: dict[str, Scorer]) -> float:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 1.0 if expected == actual else 0.0
        keys = set(expected) | set(actual)
        if not keys:
            return 1.0
        total = sum(self._similarity(expected.get(k), actual.get(k), scorers.get(k)) for k in keys)
        return total / len(keys)

    @staticmethod
    def _apply(scorer: Scorer, expected: Any, actual: Any) -> float:
        if callable(scorer) and not isinstance(scorer, (str, Metric)):
            return float(scorer(actual, expected))
        metric = resolve_metric(scorer)
        assert isinstance(metric, Metric)  # a scorer compares values via score()
        result = metric.score(actual, expected)
        assert not isinstance(result, dict)  # element scorers return a scalar
        return float(result)

    @staticmethod
    def _default_scorer(expected: Any, actual: Any) -> FieldMetric:
        """Type-aware default similarity metric for a pair of scalar values.

        ``bool`` → exact, number → graded :class:`NumericCloseness`, ``str`` →
        :class:`Fuzzy` (or exact without rapidfuzz), everything else → exact.
        """
        if isinstance(expected, bool) or isinstance(actual, bool):
            return ExactMatch()
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return NumericCloseness()
        if isinstance(expected, str) and isinstance(actual, str) and _HAS_RAPIDFUZZ:
            from structured_eval.metrics.fuzzy import Fuzzy

            return Fuzzy()
        return ExactMatch()
