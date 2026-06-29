"""TokenF1 — SQuAD-style multiset token-overlap F1.

Punctuation is dropped; repeated tokens count with multiplicity. String-only.
"""

import pytest

from structured_eval import TokenF1

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("the quick fox", "the quick fox", 1.0),  # identical
        # {the,quick,brown,fox} vs {the,quick,fox}: p=3/4, r=1 → 2·.75·1/1.75
        ("the quick brown fox", "the quick fox", 2 * 0.75 * 1.0 / 1.75),
        ("alpha", "beta", 0.0),  # disjoint
        ("", "", 1.0),  # both empty
        ("", "x", 0.0),  # one empty
        ("hello, world.", "hello world", 1.0),  # punctuation ignored
        # multiset: "the the cat"(3) vs "the cat"(2), common 2 → p=2/3,r=1 → 0.8
        ("the the cat", "the cat", 0.8),
    ],
    ids=["identical", "partial", "disjoint", "both-empty", "one-empty", "punct", "multiset"],
)
def test_score(actual, expected, score) -> None:
    assert TokenF1().score(actual, expected) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(None, None), (None, "none"), (123, 123.0)],
    ids=["null-null", "null-str", "int-float"],
)
def test_string_only_non_str_is_zero(actual, expected) -> None:
    assert TokenF1().score(actual, expected) == 0.0
