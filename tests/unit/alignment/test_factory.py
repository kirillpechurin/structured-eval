"""make_aligner — build the right aligner from a strategy + params dict."""

import pytest

from structured_eval.alignment import (
    ByIndexAligner,
    ByKeyAligner,
    HungarianAligner,
    make_aligner,
)
from structured_eval.model.config import ArrayStrategy

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("strategy", "params", "expected_type"),
    [
        (ArrayStrategy.BY_INDEX, None, ByIndexAligner),
        (ArrayStrategy.HUNGARIAN, {"threshold": 0.9}, HungarianAligner),
        (ArrayStrategy.BY_KEY, {"key": "id"}, ByKeyAligner),
        # a key_metric given as a registered name string is resolved to an instance
        (ArrayStrategy.BY_KEY, {"key": "id", "key_metric": "exact_match"}, ByKeyAligner),
    ],
    ids=["by-index", "hungarian", "by-key", "by-key-named-metric"],
)
def test_builds_expected_aligner(strategy, params, expected_type):
    assert isinstance(make_aligner(strategy, params), expected_type)
