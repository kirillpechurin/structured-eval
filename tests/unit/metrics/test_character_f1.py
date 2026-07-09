"""CharacterF1 — multiset character-overlap F1 for short free-text fields.

String-only: any non-``str`` side scores 0.0. By default punctuation and
whitespace are dropped and characters are lowercased before the multiset
comparison; each step is toggled by ``ignore_case`` / ``ignore_whitespace`` /
``ignore_punctuation``.
"""

from typing import Any

import pytest

from structured_eval.metrics import CharacterF1

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


@pytest.mark.parametrize(
    ("kwargs", "actual", "expected", "score"),
    [
        # case-sensitive: "AB" vs "ab" share nothing
        ({"ignore_case": False}, "AB", "ab", 0.0),
        # case-sensitive: only the shared "b" survives → p=r=1/2
        ({"ignore_case": False}, "Ab", "ab", 0.5),
        ({"ignore_case": True}, "AB", "ab", 1.0),
        # punctuation kept: "," counts toward the multiset → p=3/4, r=3/3
        ({"ignore_punctuation": False}, "a,bc", "abc", 6 / 7),
        ({"ignore_punctuation": False}, "a,b", "a,b", 1.0),
        # whitespace kept: the space counts → p=3/4, r=3/3
        ({"ignore_whitespace": False}, "a bc", "abc", 6 / 7),
        ({"ignore_whitespace": False}, "a b", "a b", 1.0),
        # combined: nothing normalized at all → verbatim character multiset
        (
            {
                "ignore_case": False,
                "ignore_punctuation": False,
                "ignore_whitespace": False,
            },
            "A, b",
            "a,b",
            # actual={A,',',' ',b}, expected={a,',',b} → same={',',b}=2
            2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3),
        ),
        (
            {
                "ignore_case": False,
                "ignore_punctuation": False,
                "ignore_whitespace": False,
            },
            "A, b",
            "A, b",
            1.0,
        ),
        # punctuation-only strings stay non-empty when punctuation is kept
        ({"ignore_punctuation": False}, "!!!", "!!!", 1.0),
        # whitespace kept but punctuation dropped: only the space remains
        ({"ignore_whitespace": False}, " ", "!", 0.0),
    ],
    ids=[
        "case-sensitive-disjoint",
        "case-sensitive-partial",
        "case-insensitive",
        "punct-kept-partial",
        "punct-kept-identical",
        "space-kept-partial",
        "space-kept-identical",
        "no-normalization-partial",
        "no-normalization-identical",
        "punct-only-kept",
        "space-vs-punct",
    ],
)
def test_normalization_toggles(
    kwargs: dict[str, bool], actual: str, expected: str, score: float
) -> None:
    assert CharacterF1(**kwargs).score(actual, expected) == pytest.approx(score)
