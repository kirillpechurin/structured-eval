"""HungarianAligner — optimal one-to-one assignment (scipy).

Owns its element-similarity logic and a ``Scorer`` type: a single
metric/name/callable, or a per-field ``dict[str, Scorer]``. ``key`` picks the
field(s) compared — one, or several as a composite key — and ``scorer`` how.
"""

from typing import Any

import pytest

from structured_eval.alignment import HungarianAligner
from structured_eval.models.config import ArrayStrategy

pytestmark = pytest.mark.unit


def test_strategy_recorded() -> None:
    assert HungarianAligner().align([1], [1]).strategy == ArrayStrategy.HUNGARIAN


def test_optimal_reorder() -> None:
    # numeric closeness pairs equal values regardless of order
    r = HungarianAligner().align([10, 20, 30], [30, 10, 20])
    assert sorted(r.matched) == [(0, 1), (1, 2), (2, 0)]
    assert r.missed == []
    assert r.spurious == []


def test_below_threshold_unmatched() -> None:
    # 100 vs 5 similarity is far below 0.8 → missed + spurious, no match
    r = HungarianAligner(threshold=0.8).align([100], [5])
    assert r.matched == []
    assert r.missed == [0]
    assert r.spurious == [0]


def test_empty_actual_side() -> None:
    r = HungarianAligner().align([1, 2], [])
    assert r.matched == []
    assert r.missed == [0, 1]
    assert r.spurious == []


@pytest.mark.parametrize(
    ("expected", "actual", "matched"),
    [
        # Numbers are graded by NumericCloseness: 1 - 1/9 = 0.89 clears 0.8 ...
        ([9], [8], [(0, 0)]),
        # ... while 1 - 4/9 = 0.56 does not.
        ([9], [5], []),
        # bool is an int subclass, so without the explicit bool branch these
        # would reach NumericCloseness — which rejects bools (True is not 1) and
        # would score even equal ones 0.0, leaving them unmatched.
        ([True, False], [True, False], [(0, 0), (1, 1)]),
        ([True], [False], []),
        # Strings fall through to ExactMatch, so a near miss does not align even
        # though Fuzzy("pythn", "python") would score ~0.91 and clear 0.8.
        (["python"], ["python"], [(0, 0)]),
        (["python"], ["pythn"], []),
    ],
    ids=[
        "number-near",
        "number-far",
        "bool-equal",
        "bool-differ",
        "str-equal",
        "str-near-miss",
    ],
)
def test_default_scorer_is_type_aware(
    expected: list[Any], actual: list[Any], matched: list[tuple[int, int]]
) -> None:
    assert HungarianAligner().align(expected, actual).matched == matched


def test_dict_elements_mean_agreement() -> None:
    expected = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    actual = [{"id": "b", "v": 2}, {"id": "a", "v": 1}]
    r = HungarianAligner(threshold=0.9).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_scorer_on_key_field() -> None:
    expected = [{"id": "acme"}, {"id": "globex"}]
    actual = [{"id": "globex"}, {"id": "acme"}]
    r = HungarianAligner(key="id", threshold=0.6).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_per_field_scorer_dict() -> None:
    # a dict scorer compares arrays of objects field by field
    expected = [{"name": "acme", "amount": 100}, {"name": "globex", "amount": 200}]
    actual = [{"name": "globe", "amount": 205}, {"name": "acme", "amount": 99}]
    r = HungarianAligner(
        scorer={"name": "fuzzy", "amount": "numeric_closeness"}, threshold=0.7
    ).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    ("expected", "actual", "matched", "missed", "spurious"),
    [
        pytest.param(
            [{"sku": "A", "wh": "east"}, {"sku": "A", "wh": "west"}],
            [{"sku": "A", "wh": "west"}, {"sku": "A", "wh": "east"}],
            [(0, 1), (1, 0)],
            [],
            [],
            id="shared-sku-split-by-warehouse",  # a lone sku key would collide these
        ),
        pytest.param(
            [{"sku": "A", "wh": "east"}],
            [{"sku": "A", "wh": "west"}],
            [],
            [0],
            [0],
            id="one-field-differs-no-match",  # mean 0.5 < threshold 0.8
        ),
    ],
)
def test_composite_key(
    expected: list[Any],
    actual: list[Any],
    matched: list[tuple[int, int]],
    missed: list[int],
    spurious: list[int],
) -> None:
    r = HungarianAligner(key=["sku", "wh"]).align(expected, actual)
    assert sorted(r.matched) == matched
    assert r.missed == missed
    assert r.spurious == spurious


def test_composite_key_binds_scorer_per_key_field() -> None:
    # "wh" has no entry → type default (exact); the fuzzy sku carries the pair,
    # mean (0.9 + 1.0) / 2 clears 0.9, while a differing warehouse would not.
    expected = [{"sku": "widget-a", "wh": "east"}]
    actual = [{"sku": "widget-b", "wh": "east"}]
    r = HungarianAligner(key=["sku", "wh"], scorer={"sku": "fuzzy"}, threshold=0.9)
    assert r.align(expected, actual).matched == [(0, 0)]
    assert r.align(expected, [{"sku": "widget-b", "wh": "west"}]).matched == []


def test_composite_key_single_scorer_applies_to_each_field() -> None:
    expected = [{"first": "Jonathan", "last": "Smith"}]
    actual = [{"first": "John", "last": "Smyth"}]
    r = HungarianAligner(key=["first", "last"], scorer="fuzzy", threshold=0.7).align(
        expected, actual
    )
    assert r.matched == [(0, 0)]  # mean of the two fuzzy scores clears 0.7


def test_composite_key_takes_nested_paths() -> None:
    expected = [{"who": {"first": "Jonathan"}, "qty": 1}]
    actual = [{"who": {"first": "John"}, "qty": 99}]
    # qty is not a key field, so its mismatch cannot sink the pair
    r = HungarianAligner(
        key=["who.first"], scorer={"who.first": "fuzzy"}, threshold=0.6
    ).align(expected, actual)
    assert r.matched == [(0, 0)]


def test_single_key_list_matches_string_key() -> None:
    expected = [{"id": "acme"}, {"id": "globex"}]
    actual = [{"id": "globex"}, {"id": "acme"}]
    assert HungarianAligner(key=["id"], threshold=0.6).align(
        expected, actual
    ) == HungarianAligner(key="id", threshold=0.6).align(expected, actual)


def test_empty_key_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        HungarianAligner(key=[])


def test_scorer_field_outside_key_rejected() -> None:
    with pytest.raises(ValueError, match=r"\['skuu'\] that are not in key"):
        HungarianAligner(key=["sku", "wh"], scorer={"sku": "fuzzy", "skuu": "fuzzy"})
