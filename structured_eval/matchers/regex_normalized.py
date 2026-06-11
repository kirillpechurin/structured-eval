from __future__ import annotations

import re
from typing import Any

from structured_eval.matchers.protocol import MatcherBase


class RegexNormalizedMatcher(MatcherBase):
    """Equality after regex-based normalization.

    Both values are lowercased and stripped, then every match of ``pattern``
    is replaced with ``repl`` before comparing. The default pattern collapses
    whitespace runs, giving "lowercase + strip + collapse spaces".

    Inject a different pattern to control what is ignored, e.g.::

        RegexNormalizedMatcher(pattern=r"[^\\w\\s]", repl="")  # drop punctuation
        RegexNormalizedMatcher(pattern=r"[-_]", repl=" ")       # dashes → spaces
    """

    name = "NORMALIZED"

    def __init__(self, pattern: str | re.Pattern[str] = r"\s+", repl: str = " "):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.repl = repl

    def _normalize(self, value: Any) -> str:
        return self.pattern.sub(self.repl, str(value).lower().strip()).strip()

    def similarity(self, actual: Any, expected: Any) -> float:
        return 1.0 if self._normalize(actual) == self._normalize(expected) else 0.0
