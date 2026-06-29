"""CharacterF1 — multiset character-overlap F1 for short free-text fields.

String-only: any non-``str`` side scores 0.0. Punctuation and whitespace are
dropped and characters are lowercased before the multiset comparison.
"""

from typing import Any

import pytest

from structured_eval import CharacterF1

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("abc", "abc", 1.0),  # identical
        ("", "", 1.0),  # both empty → vacuously 1.0
        ("Hello", "hello", 1.0),  # case-insensitive
        ("a, b, c", "abc", 1.0),  # punctuation/whitespace dropped
        ("abc", "xyz", 0.0),  # no shared characters
        ("aabb", "ab", 2 / 3),  # multiset: p=2/4, r=2/2 → F1=2/3
    ],
    ids=["identical", "both-empty", "case", "punct-stripped", "disjoint", "multiset"],
)
def test_character_f1(actual: Any, expected: Any, score: Any) -> None:
    assert CharacterF1().score(actual, expected) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("abc", ""),  # one side empty after normalization
        ("!!!", "abc"),  # normalizes to empty → no overlap
        (123, "123"),  # non-str actual
        ("123", 123),  # non-str expected
        (None, None),  # null → 0.0 (no equality fallback)
    ],
    ids=["one-empty", "punct-only", "int-actual", "int-expected", "null"],
)
def test_non_overlap_or_non_string_is_zero(actual: Any, expected: Any) -> None:
    assert CharacterF1().score(actual, expected) == 0.0
