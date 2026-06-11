from __future__ import annotations

from typing import Any

from structured_eval.matchers.protocol import MatcherBase


class ExactMatcher(MatcherBase):
    """Exact equality: ``actual == expected``."""

    name = "EXACT"

    def similarity(self, actual: Any, expected: Any) -> float:
        return 1.0 if actual == expected else 0.0
