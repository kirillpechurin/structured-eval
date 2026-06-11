from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

from structured_eval.matchers.protocol import MatcherBase


def _normalize_url(url: Any) -> str:
    """Canonicalize scheme/host case, trailing slash, and query order."""
    try:
        parsed = urlparse(str(url).strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        query = urlencode(sorted(parse_qsl(parsed.query)))
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return str(url).lower().strip()


class UrlMatcher(MatcherBase):
    """Exact equality after URL normalization."""

    name = "URL"

    def similarity(self, actual: Any, expected: Any) -> float:
        return 1.0 if _normalize_url(actual) == _normalize_url(expected) else 0.0
