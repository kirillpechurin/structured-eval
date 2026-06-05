from __future__ import annotations

# TODO: refactor to class-per-matcher architecture with a shared Protocol/ABC
# (e.g. class Matcher(Protocol): def score(self, actual, expected) -> float).
# Current dispatch-function approach works for MVP but is harder to extend:
# adding a new matcher requires touching match(), and there's no natural place
# for per-matcher state or configuration beyond what the dataclass carries.
# It is possible to use Generic[MatcherType] here, but it will add an additional
# complexity to user matchers. I should rethink this module.
import re
from typing import Any

from structured_eval.core.config import (
    MatcherType,
    MatchMode,
    NumericMode,
    _Custom,
    _Exact,
    _Fuzzy,
    _Normalized,
    _Numeric,
    _TokenF1,
)

# ── Exact ─────────────────────────────────────────────────────────────────────


def _match_exact(actual: Any, expected: Any) -> float:
    return 1.0 if actual == expected else 0.0


# ── Normalized ────────────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def _normalize(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value).lower().strip())


def _match_normalized(actual: Any, expected: Any) -> float:
    return 1.0 if _normalize(actual) == _normalize(expected) else 0.0


# ── Numeric ───────────────────────────────────────────────────────────────────


def _match_numeric(actual: Any, expected: Any, tolerance: float, mode: NumericMode) -> float:
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return 0.0

    if mode == NumericMode.RELATIVE:
        deviation = 0.0 if e == 0 and a == 0 else (abs(a - e) / abs(e) if e != 0 else float("inf"))
    else:
        deviation = abs(a - e)

    if tolerance == 0:
        return 1.0 if deviation == 0 else 0.0
    return max(0.0, 1.0 - deviation / tolerance)


# ── Token F1 ──────────────────────────────────────────────────────────────────

_NON_WORD = re.compile(r"[^\w\s]")


def _tokenize(value: Any) -> list[str]:
    return _NON_WORD.sub(" ", str(value).lower()).split()


def _match_token_f1(actual: Any, expected: Any) -> float:
    actual_tokens = set(_tokenize(actual))
    expected_tokens = set(_tokenize(expected))

    if not actual_tokens and not expected_tokens:
        return 1.0
    if not actual_tokens or not expected_tokens:
        return 0.0

    intersection = len(actual_tokens & expected_tokens)
    precision = intersection / len(actual_tokens)
    recall = intersection / len(expected_tokens)

    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


# ── Fuzzy ─────────────────────────────────────────────────────────────────────


def _match_fuzzy(actual: Any, expected: Any) -> float:
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rapidfuzz is required for FUZZY matching. "
            "Install it with: pip install 'structured-eval[fuzzy]'"
        ) from exc
    # TODO: token_sort_ratio may not suit all cases (short codes, structured strings,
    # numeric-heavy text). Consider exposing the fuzz algorithm as a _Fuzzy parameter.
    return fuzz.token_sort_ratio(str(actual), str(expected)) / 100.0


# ── Custom ────────────────────────────────────────────────────────────────────


def _match_custom(actual: Any, expected: Any, matcher: _Custom) -> float:
    result = matcher.fn(actual, expected)
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    return float(result)


# ── Dispatcher ────────────────────────────────────────────────────────────────


def match(matcher: MatcherType, actual: Any, expected: Any) -> float:
    """Apply matcher to (actual, expected) and return a score in [0.0, 1.0]."""
    if isinstance(matcher, _Exact):
        return _match_exact(actual, expected)
    if isinstance(matcher, _Normalized):
        return _match_normalized(actual, expected)
    if isinstance(matcher, _Numeric):
        return _match_numeric(actual, expected, matcher.tolerance, matcher.mode)
    if isinstance(matcher, _TokenF1):
        return _match_token_f1(actual, expected)
    if isinstance(matcher, _Fuzzy):
        return _match_fuzzy(actual, expected)
    if isinstance(matcher, _Custom):
        return _match_custom(actual, expected, matcher)
    raise TypeError(f"Unknown matcher type: {type(matcher)!r}")


# ── Auto-detect ───────────────────────────────────────────────────────────────

# TODO: name-based heuristic is too coarse — "country_description" incorrectly maps to EXACT
# because "country" appears as a token. Needs a proper design: suffix-only matching,
# an explicit allow-list, or user-configurable rules before this is production-ready.
_EXACT_NAME = re.compile(
    r"(^|_)(id|code|status|type|kind|currency|country|language|locale|flag|mode|state)(_|$)",
    re.IGNORECASE,
)


def detect_matcher(field_name: str, expected_value: Any) -> MatcherType:
    """Infer a matcher from field name and expected value type.

    Priority order:
      1. Name matches id/code/status/... → EXACT
      2. Expected is a number (int/float, not bool) → NUMERIC
      3. Everything else → TOKEN_F1
    """
    if _EXACT_NAME.search(field_name):
        return MatchMode.EXACT
    if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
        return MatchMode.NUMERIC()
    return MatchMode.TOKEN_F1
