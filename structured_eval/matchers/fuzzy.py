from __future__ import annotations

from typing import Any

from structured_eval.matchers.protocol import MatcherBase


class FuzzyMatcher(MatcherBase):
    """Fuzzy string similarity via RapidFuzz (optional dependency)."""

    name = "FUZZY"

    def similarity(self, actual: Any, expected: Any) -> float:
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rapidfuzz is required for FUZZY matching. "
                "Install it with: pip install 'structured-eval[fuzzy]'"
            ) from exc
        return fuzz.token_sort_ratio(str(actual), str(expected)) / 100.0
