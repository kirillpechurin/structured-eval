from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from structured_eval.metrics.base import FieldMetric


class UrlMatch(FieldMetric):
    """Equivalence match for URL fields after normalization.

    Two URLs score ``1.0`` when they are equivalent once cosmetically different
    but meaning-preserving components are normalized, and ``0.0`` otherwise.

    Normalization applied to each side:

    - **scheme** and **host** are lowercased;
    - a leading ``www.`` on the host is stripped (unless ``ignore_www=False``);
    - the **path** is percent-decoded and a trailing slash is normalized away;
    - **query** parameters are percent-decoded and sorted, so parameter order
      does not matter (dropped entirely when ``ignore_query=True``);
    - the **fragment** is dropped when ``ignore_fragment=True`` (the default).

    So ``https://Example.com`` and ``https://example.com/`` are equivalent, and
    URLs differing only in query-parameter order match, while different paths or
    hosts do not.

    Both sides must be non-empty strings that parse to a URL with a scheme and a
    host. Anything else — a non-string, an empty string, or a bare path with no
    scheme/host — scores ``0.0``.
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
        super().__init__(name=name)
        self.ignore_query = ignore_query
        self.ignore_fragment = ignore_fragment
        self.ignore_www = ignore_www

    def _normalize(self, value: Any) -> tuple[str, ...] | None:
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
        norm_actual = self._normalize(actual)
        norm_expected = self._normalize(expected)
        if norm_actual is None or norm_expected is None:
            return 0.0
        return 1.0 if norm_actual == norm_expected else 0.0
