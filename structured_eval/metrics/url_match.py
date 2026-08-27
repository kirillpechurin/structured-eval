"""The `url_match` metric — URL equivalence after normalization."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null


class UrlMatch(FieldMetric):
    """Equivalence match for URL fields after normalization.

    Normalization applied to each side before comparing:

    - scheme and host lowercased, and a leading `www.` stripped unless
      `ignore_www=False`;
    - path percent-decoded, a trailing slash normalized away;
    - query parameters percent-decoded and sorted, so their order is
      irrelevant — or dropped entirely when `ignore_query=True`;
    - fragment dropped when `ignore_fragment=True`, which is the default.

    Both sides must parse to a URL with a scheme and a host; anything else
    scores 0.0. Two `None`s are the exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import UrlMatch
        >>> from structured_eval.models import EvalConfig
        >>> UrlMatch().score("https://Example.com/a/", "https://example.com/a")
        1.0
        >>> UrlMatch().score("https://example.com/a", "https://example.com/b")
        0.0
        >>> UrlMatch().score("https://example.com/a?x=1", "https://example.com/a")
        0.0
        >>> UrlMatch(ignore_query=True).score("https://example.com/a?x=1",
        ...                                   "https://example.com/a")
        1.0
        >>> report = evaluate({"site": "https://WWW.Example.com/docs/"},
        ...                   {"site": "https://example.com/docs"},
        ...                   EvalConfig(metrics=[UrlMatch()]))
        >>> float(report.field_scores["site"].metrics["url_match"])
        1.0
    """

    name = "url_match"

    def __init__(
        self,
        *,
        ignore_query: bool = False,
        ignore_fragment: bool = True,
        ignore_www: bool = True,
        name: str | None = None,
    ) -> None:
        """Choose which URL components are treated as insignificant.

        Args:
            ignore_query: Drop the query string entirely.
            ignore_fragment: Drop the `#fragment`.
            ignore_www: Treat `www.host` and `host` as the same.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.ignore_query = ignore_query
        self.ignore_fragment = ignore_fragment
        self.ignore_www = ignore_www

    def _normalize(self, value: Any) -> tuple[str, ...] | None:
        """The URL's comparable parts, or `None` when it is not a URL at all."""
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parts = urlsplit(value.strip())
        except ValueError:
            return None
        if not parts.scheme or not parts.hostname:
            return None

        scheme = parts.scheme.lower()
        host = parts.hostname.lower()
        if self.ignore_www and host.startswith("www."):
            host = host[4:]

        netloc = f"{host}:{parts.port}" if parts.port is not None else host

        path = unquote(parts.path)
        if path.endswith("/"):
            path = path[:-1]

        if self.ignore_query:
            query = ""
        else:
            pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
            query = "&".join(f"{k}={v}" for k, v in pairs)

        fragment = "" if self.ignore_fragment else unquote(parts.fragment)

        return (urlunsplit((scheme, netloc, path, query, fragment)),)

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when both URLs normalize to the same form, else 0.0."""
        if both_null(actual, expected):
            return 1.0
        norm_actual = self._normalize(actual)
        norm_expected = self._normalize(expected)
        if norm_actual is None or norm_expected is None:
            return 0.0
        return 1.0 if norm_actual == norm_expected else 0.0
