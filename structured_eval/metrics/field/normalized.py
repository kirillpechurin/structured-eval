from __future__ import annotations

import re
from typing import Any

from structured_eval.metrics.protocol import FieldMetric


class NormalizedMatch(FieldMetric):
    """Equality after regex-based normalization → 1.0, else 0.0.

    Both values are lowercased and stripped, then every match of ``pattern`` is
    replaced with ``repl`` before comparing. The default collapses whitespace
    runs ("lowercase + strip + collapse spaces"). Inject a pattern to control
    what is ignored, e.g.::

        NormalizedMatch(pattern=r"[^\\w\\s]", repl="")  # drop punctuation
        NormalizedMatch(pattern=r"[-_]", repl=" ")       # dashes → spaces
    """

    name = "normalized_match"

    def __init__(self, pattern: str | re.Pattern[str] = r"\s+", repl: str = " "):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.repl = repl

    def _normalize(self, value: Any) -> str:
        return self.pattern.sub(self.repl, str(value).lower().strip()).strip()

    def score(self, actual: Any, expected: Any) -> float:
        return 1.0 if self._normalize(actual) == self._normalize(expected) else 0.0
