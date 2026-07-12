"""TokenF1 — SQuAD-style multiset token-overlap F1.

Defaults reproduce the official SQuAD ``normalize_answer`` (lowercase, drop
punctuation, drop articles); each step is an independent toggle. String-only.
"""

from typing import Any

import pytest

from structured_eval.metrics import TokenF1

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("the quick fox", "the quick fox", 1.0),  # identical
        # articles dropped: {quick,brown,fox} vs {quick,fox} → p=2/3, r=1
        ("the quick brown fox", "the quick fox", 0.8),
        ("alpha", "beta", 0.0),  # disjoint
        ("", "", 1.0),  # both empty
        ("", "x", 0.0),  # one empty
        ("hello, world.", "hello world", 1.0),  # punctuation ignored
        ("HELLO", "hello", 1.0),  # case ignored
        ("don't", "dont", 1.0),  # punctuation removed, not split on
        ("a cat and an ox", "cat and ox", 1.0),  # articles ignored
        ("the", "", 1.0),  # both normalize to empty
        # multiset: "cat cat"(2) vs "cat"(1), common 1 → p=1/2, r=1
        ("cat cat", "cat", 2 * 0.5 * 1.0 / 1.5),
        (None, None, 1.0),  # no text expected, none given → they agree
    ],
    ids=[
        "identical",
        "partial",
        "disjoint",
        "both-empty",
        "one-empty",
        "punct",
        "case",
        "apostrophe",
        "articles",
        "articles-only",
        "multiset",
        "both-null",
    ],
)
def test_score_defaults(actual: Any, expected: Any, score: Any) -> None:
    assert TokenF1().score(actual, expected) == pytest.approx(score)


def test_squad_normalize_answer_is_reproduced() -> None:
    """The reference normalization: lowercase, punctuation, articles, whitespace."""
    assert TokenF1().score("the Amazon rainforest.", "Amazon   Rainforest") == 1.0


@pytest.mark.parametrize(
    ("kwargs", "actual", "expected", "score"),
    [
        # case-sensitive: "AB" and "ab" are different tokens → no overlap
        ({"ignore_case": False}, "AB", "ab", 0.0),
        ({"ignore_case": False}, "ab", "ab", 1.0),
        # punctuation kept: it sticks to the token it touches
        ({"ignore_punctuation": False}, "hello, world.", "hello world", 0.0),
        ({"ignore_punctuation": False}, "hi!", "hi!", 1.0),
        # articles kept: they count as tokens like any other word
        ({"ignore_articles": False}, "the cat", "cat", 2 * 0.5 * 1.0 / 1.5),
        ({"ignore_articles": False}, "the the cat", "the cat", 0.8),
        # toggles are independent: articles go regardless of case-folding
        ({"ignore_case": False}, "The cat", "cat", 1.0),
        # combinations
        ({"ignore_case": False, "ignore_punctuation": False}, "Hi!", "hi!", 0.0),
        ({"ignore_case": False, "ignore_punctuation": False}, "Hi!", "Hi!", 1.0),
        # {the,fox.} vs {the,fox}: only "the" is shared → p=r=0.5
        (
            {"ignore_punctuation": False, "ignore_articles": False},
            "the fox.",
            "the fox",
            0.5,
        ),
        (
            {
                "ignore_case": False,
                "ignore_punctuation": False,
                "ignore_articles": False,
            },
            "The Fox.",
            "The Fox.",
            1.0,
        ),
    ],
    ids=[
        "case-sensitive-differs",
        "case-sensitive-equal",
        "punct-kept-differs",
        "punct-kept-equal",
        "articles-kept",
        "articles-kept-multiset",
        "articles-dropped-when-cased",
        "case-and-punct-differ",
        "case-and-punct-equal",
        "punct-and-articles",
        "all-off",
    ],
)
def test_score_toggles(kwargs: Any, actual: Any, expected: Any, score: Any) -> None:
    assert TokenF1(**kwargs).score(actual, expected) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(None, "none"), (123, 123.0)],
    ids=["null-str", "int-float"],
)
def test_string_only_non_str_is_zero(actual: Any, expected: Any) -> None:
    assert TokenF1().score(actual, expected) == 0.0
