"""The `regex_match` metric — equality after an optional regex rewrite."""

from __future__ import annotations

import re
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null


class RegexMatch(FieldMetric):
    r"""String equality after an optional regex rewrite → 1.0, else 0.0.

    Each side is lowered and stripped if asked, then every match of `pattern`
    is replaced by `repl`, and the results are compared exactly. String-only: a
    non-`str` side scores 0.0. Two `None`s are the exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import RegexMatch
        >>> from structured_eval.models import EvalConfig
        >>> RegexMatch().score("Hello   World", "hello world")
        1.0
        >>> RegexMatch(pattern=r"[^\w\s]", repl="").score("Acme, Inc.", "Acme Inc")
        1.0
        >>> drop_punctuation = RegexMatch(pattern=r"[^\w\s]", repl="")
        >>> report = evaluate({"vendor": "Acme, Inc."}, {"vendor": "Acme Inc"},
        ...                   EvalConfig(metrics=[drop_punctuation]))
        >>> float(report.field_scores["vendor"].metrics["regex_match"])
        1.0
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
        """Set the rewrite applied to both sides before comparing.

        Args:
            pattern: What to substitute; the default collapses whitespace.
            repl: What to substitute it with.
            lower: Lowercase both sides first.
            strip: Trim both ends, before and after the substitution.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.repl = repl
        self.lower = lower
        self.strip = strip

    def _normalize(self, value: str) -> str:
        """The string as this instance rewrites it before comparing."""
        if self.lower:
            value = value.lower()
        if self.strip:
            value = value.strip()
        value = self.pattern.sub(self.repl, value)
        return value.strip() if self.strip else value

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when the two rewritten strings are equal, else 0.0."""
        if both_null(actual, expected):
            return 1.0
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0
        return 1.0 if self._normalize(actual) == self._normalize(expected) else 0.0
