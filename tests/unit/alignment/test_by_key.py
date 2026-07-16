"""ByKeyAligner — match by key, globally greedy best-first.

A whole-element key reduces to first-match; a named key reorders; a soft
(graded) key picks the strongest free partner, order-independently. A key may
also name several fields — a composite key scored as the mean over the fields.
"""

from typing import Any

import pytest

from structured_eval.alignment import ByKeyAligner
from structured_eval.models.config import ArrayStrategy

pytestmark = pytest.mark.unit


def test_whole_element_key() -> None:
    r = ByKeyAligner().align([1, 2], [2, 1])  # (expected, actual)
    # expected 1 pairs actual idx1; expected 2 pairs actual idx0
    assert sorted(r.matched) == [(0, 1), (1, 0)]
    assert r.missed == []
    assert r.spurious == []


def test_named_key_reorders() -> None:
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "b"}, {"id": "a"}]
    r = ByKeyAligner(key="id").align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_missing_and_spurious() -> None:
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "a"}, {"id": "c"}]
    r = ByKeyAligner(key="id").align(expected, actual)
    assert r.matched == [(0, 0)]
    assert r.missed == [1]  # b
    assert r.spurious == [1]  # c


def test_duplicate_keys_paired_once() -> None:
    # one expected "a", two actual "a": one pairs, the other is spurious
    r = ByKeyAligner(key="id").align([{"id": "a"}], [{"id": "a"}, {"id": "a"}])
    assert r.matched == [(0, 0)]
    assert r.spurious == [1]


def test_soft_key_picks_best_not_first() -> None:
    # both actuals clear the threshold, but pairing is greedy best-first —
    # the closer id wins, not the first one seen.
    expected = [{"id": 10}]
    actual = [{"id": 5}, {"id": 9}]  # closeness 0.5 vs 0.9
    r = ByKeyAligner(key="id", key_metric="numeric_closeness", threshold=0.4).align(
        expected, actual
    )
    assert r.matched == [(0, 1)]  # 9 (best), not 5 (first)
    assert r.spurious == [0]


def test_soft_key_global_greedy_is_order_independent() -> None:
    # two expected each best-match a different actual; global best-first resolves
    # the assignment without an order bias.
    expected = [{"id": 10}, {"id": 8}]
    actual = [{"id": 8}, {"id": 10}]
    r = ByKeyAligner(key="id", key_metric="numeric_closeness", threshold=0.5).align(
        expected, actual
    )
    assert sorted(r.matched) == [(0, 1), (1, 0)]  # 10↔10, 8↔8
    assert r.missed == []
    assert r.spurious == []


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
            id="one-field-differs-no-match",  # mean 0.5 < threshold 1.0
        ),
        pytest.param(
            [{"sku": "A", "wh": "east"}],
            [{"sku": "A", "wh": "east"}, {"sku": "A", "wh": "east"}],
            [(0, 0)],
            [],
            [1],
            id="tie-breaks-by-order",  # equal keys → first actual claimed
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
    r = ByKeyAligner(key=["sku", "wh"]).align(expected, actual)
    assert r.matched == matched
    assert r.missed == missed
    assert r.spurious == spurious


def test_composite_key_soft_metric_means_fields() -> None:
    # id 10 vs 9 → 0.9 and lot 100 vs 100 → 1.0, mean 0.95, clears 0.9; the
    # other candidate (id 10 vs 5 → 0.5, mean 0.75) does not.
    expected = [{"id": 10, "lot": 100}]
    actual = [{"id": 5, "lot": 100}, {"id": 9, "lot": 100}]
    r = ByKeyAligner(
        key=["id", "lot"], key_metric="numeric_closeness", threshold=0.9
    ).align(expected, actual)
    assert r.matched == [(0, 1)]
    assert r.spurious == [0]


def test_single_key_list_matches_string_key() -> None:
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "b"}, {"id": "c"}]
    assert ByKeyAligner(key=["id"]).align(expected, actual) == ByKeyAligner(
        key="id"
    ).align(expected, actual)


def test_empty_key_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        ByKeyAligner(key=[])


def test_strategy_recorded() -> None:
    assert ByKeyAligner().align([], []).strategy == ArrayStrategy.BY_KEY
