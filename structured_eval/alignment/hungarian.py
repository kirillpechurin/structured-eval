from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from structured_eval.alignment.base import ArrayAligner, key_value, normalize_key
from structured_eval.metrics.base import FieldMetric, Metric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.metrics.numeric_closeness import NumericCloseness
from structured_eval.models.config import ArrayStrategy
from structured_eval.models.nodes.array_node import ArrayMatchResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    * ``None`` — type-aware default (graded numeric, exact for everything else),
      with objects scored field-by-field.

    ``key`` scores on named field(s) instead of the whole element: one field
    path, or several — ``["sku", "warehouse"]`` — for records identified by a
    combination. Paths may be nested (``"who.first"``). ``key`` picks *what* is
    compared and ``scorer`` *how*: with ``key`` set, a ``dict`` scorer binds a
    scorer per key field (naming a field outside ``key`` is an error), a single
    scorer applies to each key field, and the element score is the mean over
    the key fields. Requires the ``align`` extra (scipy).
    """

    def __init__(
        self,
        scorer: Scorer | dict[str, Scorer] | None = None,
        threshold: float = 0.8,
        key: str | Sequence[str] | None = None,
    ):
        self.scorer = scorer
        self.threshold = threshold
        # One key or many, ``self.key`` is a list of field paths from here on
        # (``None`` keeps its meaning: score on the whole element).
        self.key = normalize_key(key, self.__class__.__name__)
        if self.key is not None and isinstance(scorer, dict):
            unknown = [field for field in scorer if field not in self.key]
            if unknown:
                raise ValueError(
                    f"HungarianAligner: scorer names field(s) {sorted(unknown)} that "
                    f"are not in key {self.key}; a scorer only tunes a key field."
                )

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

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scipy is required for HungarianAligner. "
                "Install it with: pip install 'structured-eval[align]'"
            ) from exc

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
        if self.key is None:
            return self._similarity(expected, actual, self.scorer)
        # ``key`` picks the fields to compare on, ``scorer`` says how: a dict
        # binds a scorer per key field, a single scorer applies to each of them.
        scorers = (
            self.scorer
            if isinstance(self.scorer, dict)
            else dict.fromkeys(self.key, self.scorer)
        )
        return self._object_similarity(
            {field: key_value(expected, field) for field in self.key},
            {field: key_value(actual, field) for field in self.key},
            scorers,
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

    def _object_similarity(
        self, expected: Any, actual: Any, scorers: Mapping[str, Scorer | None]
    ) -> float:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 1.0 if expected == actual else 0.0
        keys = set(expected) | set(actual)
        if not keys:
            return 1.0
        total = sum(
            self._similarity(expected.get(k), actual.get(k), scorers.get(k))
            for k in keys
        )
        return total / len(keys)

    @staticmethod
    def _apply(scorer: Scorer, expected: Any, actual: Any) -> float:
        if callable(scorer) and not isinstance(scorer, (str, Metric)):
            return float(scorer(actual, expected))
        return MetricInvoker(resolve_metric(scorer)).scalar_on_values(actual, expected)

    @staticmethod
    def _default_scorer(expected: Any, actual: Any) -> FieldMetric:
        """Type-aware default similarity metric for a pair of scalar values.

        Number → graded :class:`NumericCloseness`; everything else, ``bool`` and
        ``str`` included, → :class:`ExactMatch`. Strings are *not* graded by
        default: pass ``scorer="fuzzy"`` to pair them by similarity.
        """
        if isinstance(expected, bool) or isinstance(actual, bool):
            return ExactMatch()
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return NumericCloseness()
        return ExactMatch()
