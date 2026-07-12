from __future__ import annotations

import re
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null


class RegexMatch(FieldMetric):
    """String equality after an optional regex rewrite → 1.0, else 0.0.

    A **string-only** metric: if either side is not a ``str`` the score is
    ``0.0`` (use ``Numeric`` for numbers, ``ExactMatch`` for verbatim
    equality) — except two ``None``s, which agree (``1.0``; see
    ``metrics.utils.null``). For two strings it applies, in order, optional ``lower`` and
    ``strip``, then substitutes every match of ``pattern`` with ``repl``, and
    compares the results exactly.

    The default ``pattern=r"\\s+", repl=" "`` (with ``lower``/``strip`` on)
    collapses whitespace and ignores casing. Tune the rewrite, e.g.::

        RegexMatch(pattern=r"[^\\w\\s]", repl="")  # drop punctuation
        RegexMatch(pattern=r"[-_]", repl=" ")       # dashes/underscores → spaces
        RegexMatch(lower=False)                      # case-sensitive
    """

    name = "regex_match"

    def __init__(
        self,
        pattern: str | re.Pattern[str] = r"\s+",
        repl: str = " ",
        lower: bool = True,
        strip: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.repl = repl
        self.lower = lower
        self.strip = strip

    def _normalize(self, value: str) -> str:
        if self.lower:
            value = value.lower()
        if self.strip:
            value = value.strip()
        value = self.pattern.sub(self.repl, value)
        return value.strip() if self.strip else value

    def score(self, actual: Any, expected: Any) -> float:
        if both_null(actual, expected):
            return 1.0
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0
        return 1.0 if self._normalize(actual) == self._normalize(expected) else 0.0
