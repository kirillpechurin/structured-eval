"""UrlMatch — binary URL equivalence after equivalence-preserving normalization.

Normalizes scheme/host casing, ``www.``, default ports, trailing slash and
query-parameter order; equivalent URLs score 1.0, different ones 0.0. Non-URL
or unparseable inputs score 0.0. Strictness is tunable via ``ignore_query`` /
``ignore_fragment`` / ``ignore_www``.
"""

from typing import Any

import pytest

from structured_eval.metrics import UrlMatch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("https://example.com", "https://example.com", 1.0),  # identical
        ("https://Example.com", "https://example.com/", 1.0),  # case + slash
        ("https://example.com:443/", "https://example.com", 1.0),  # default port
        ("https://www.example.com", "https://example.com", 1.0),  # www stripped
        (
            "https://example.com/?b=2&a=1",
            "https://example.com/?a=1&b=2",
            1.0,
        ),  # query order
        ("https://example.com#top", "https://example.com", 1.0),  # fragment ignored
        ("https://example.com/a", "https://example.com/b", 0.0),  # different path
        ("https://a.com", "https://b.com", 0.0),  # different host
        ("http://example.com", "https://example.com", 0.0),  # different scheme
        ("https://example.com?a=1", "https://example.com?a=2", 0.0),  # query value
        ("https://example.com:8080", "https://example.com", 0.0),  # non-default port
    ],
    ids=[
        "identical",
        "case-slash",
        "default-port",
        "www",
        "query-order",
        "fragment",
        "diff-path",
        "diff-host",
        "diff-scheme",
        "query-value",
        "non-default-port",
    ],
)
def test_url_match(actual: Any, expected: Any, score: float) -> None:
    assert UrlMatch().score(actual, expected) == score


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("not a url", "https://example.com"),  # bare text, no scheme/host
        ("/only/a/path", "https://example.com"),  # relative path
        ("", "https://example.com"),  # empty string
        (None, "https://example.com"),  # null
        (42, "https://example.com"),  # non-string type
    ],
    ids=["text", "relative", "empty", "null", "non-string"],
)
def test_non_url_is_zero(actual: Any, expected: Any) -> None:
    assert UrlMatch().score(actual, expected) == 0.0


def test_ignore_query() -> None:
    metric = UrlMatch(ignore_query=True)
    assert metric.score("https://example.com?a=1", "https://example.com?a=2") == 1.0


def test_ignore_fragment_false() -> None:
    metric = UrlMatch(ignore_fragment=False)
    assert metric.score("https://example.com#a", "https://example.com#b") == 0.0
    assert metric.score("https://example.com#a", "https://example.com#a") == 1.0


def test_ignore_www_false() -> None:
    metric = UrlMatch(ignore_www=False)
    assert metric.score("https://www.example.com", "https://example.com") == 0.0
