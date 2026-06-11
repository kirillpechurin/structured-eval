from __future__ import annotations

import re
from typing import Any

from structured_eval.matchers.exact import ExactMatcher
from structured_eval.matchers.numeric import NumericMatcher
from structured_eval.matchers.protocol import Matcher
from structured_eval.matchers.token_f1 import TokenF1Matcher
from structured_eval.matchers.url import UrlMatcher

_URL_NAME = re.compile(
    r"(^|_)(url|link|href|uri|endpoint|website)(_|$)",
    re.IGNORECASE,
)

# Long-text suffixes — checked before _EXACT_NAME to prevent false EXACT
# matches like "country_description" or "status_note".
_TEXT_SUFFIX = re.compile(
    r"(^|_)(description|note|text|label|message|body|content|summary|detail|info|comment)(_|$)",
    re.IGNORECASE,
)

_EXACT_NAME = re.compile(
    r"(^|_)(id|code|status|type|kind|currency|country|language|locale|flag|mode|state)(_|$)",
    re.IGNORECASE,
)


def detect_matcher(field_name: str, expected_value: Any) -> Matcher:
    """Infer a matcher instance from a field name and its expected value.

    Priority:
      1. URL-named fields (``*_url``, ``*_link``, …) → ``UrlMatcher``
      2. Long-text suffixes (``*_description``, ``*_note``, …) → ``TokenF1Matcher``
      3. Exact-type names (``id``, ``code``, ``status``, …) → ``ExactMatcher``
      4. Numeric values → ``NumericMatcher``
      5. Everything else → ``TokenF1Matcher``
    """
    if _URL_NAME.search(field_name):
        return UrlMatcher()
    if _TEXT_SUFFIX.search(field_name):
        return TokenF1Matcher()
    if _EXACT_NAME.search(field_name):
        return ExactMatcher()
    if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
        return NumericMatcher()
    return TokenF1Matcher()
