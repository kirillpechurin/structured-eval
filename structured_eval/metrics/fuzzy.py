from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric


class Fuzzy(FieldMetric):
    """Fuzzy string similarity via RapidFuzz (optional dependency)."""

    name = "fuzzy"

    def score(self, actual: Any, expected: Any) -> float:
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rapidfuzz is required for the 'fuzzy' metric. "
                "Install it with: pip install 'structured-eval[fuzzy]'"
            ) from exc
        return fuzz.token_sort_ratio(str(actual), str(expected)) / 100.0
