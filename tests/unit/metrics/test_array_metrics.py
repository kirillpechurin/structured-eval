"""Unit tests for array metrics over a TreeBuilder-built ArrayNode.

Arrays are aligned first (by_index or by_key); each aligned element is graded by
its recursive ``element_score`` against ``threshold``. ``missed`` are FN,
``spurious`` are FP. Arrays are tested as a nested ``items`` field (root-list
input is a separate, roadmap concern), and the ArrayNode is pulled from the
built tree via ``children``.
"""

import pytest

from structured_eval import (
    ArrayAccuracy,
    ArrayCardinality,
    ArrayF1,
    ArrayFieldConfig,
    ArrayPrecision,
    ArrayPRF1,
    ArrayRecall,
    ArrayStrategy,
    EvalConfig,
    FieldConfig,
    ObjectFieldConfig,
)
from structured_eval.model.nodes.array_node import ArrayNode

pytestmark = pytest.mark.unit


def _array_node(tree_factory, actual_items, expected_items, item_cfg=None, **kwargs):
    """Build a doc {"items": [...]} and return its ArrayNode."""
    cfg = EvalConfig(fields={"items": ArrayFieldConfig(item=item_cfg, **kwargs)})
    root = tree_factory({"items": actual_items}, {"items": expected_items}, cfg)
    node = root.children["items"]
    assert isinstance(node, ArrayNode)
    return node


def test_f1_perfect_by_index(tree_factory):
    node = _array_node(tree_factory, [1, 2, 3], [1, 2, 3])
    assert ArrayF1().compute(node) == 1.0


def test_precision_recall_extra_and_missing(tree_factory):
    # actual [1,2,9] vs expected [1,2,3,4] by index:
    # aligned idx 0,1,2 (idx2: 9 vs 3 fails), missed idx 3
    node = _array_node(tree_factory, [1, 2, 9], [1, 2, 3, 4])
    # tp=2, predicted=3 aligned, expected = 3 + 1 missed = 4
    assert ArrayPrecision().compute(node) == pytest.approx(2 / 3)
    assert ArrayRecall().compute(node) == pytest.approx(2 / 4)


def test_spurious_lowers_precision(tree_factory):
    node = _array_node(tree_factory, [1, 2, 3], [1, 2])
    # aligned idx0,1 (tp=2), spurious idx2 → precision 2/3, recall 1.0
    assert ArrayPrecision().compute(node) == pytest.approx(2 / 3)
    assert ArrayRecall().compute(node) == 1.0


def test_prf1_returns_dict(tree_factory):
    node = _array_node(tree_factory, [1, 9], [1, 2])
    result = ArrayPRF1().compute(node)
    assert set(result) == {"array_precision", "array_recall", "array_f1"}


def test_accuracy_soft_mean(tree_factory):
    node = _array_node(tree_factory, [1, 9], [1, 2])
    assert ArrayAccuracy().compute(node) == pytest.approx(0.5)


def test_accuracy_missed_counts_zero(tree_factory):
    node = _array_node(tree_factory, [1], [1, 2])
    assert ArrayAccuracy().compute(node) == pytest.approx(0.5)


def test_cardinality(tree_factory):
    node = _array_node(tree_factory, [1, 2, 3], [1, 2])
    assert ArrayCardinality().compute(node) == pytest.approx(2 / 3)


def test_cardinality_empty_vacuous(tree_factory):
    node = _array_node(tree_factory, [], [])
    assert ArrayCardinality().compute(node) == 1.0


def test_by_key_alignment_reorders(tree_factory):
    item = ObjectFieldConfig(fields={"id": FieldConfig(), "v": FieldConfig()})
    actual = [{"id": "b", "v": 2}, {"id": "a", "v": 1}]
    expected = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    node = _array_node(
        tree_factory,
        actual,
        expected,
        item_cfg=item,
        strategy=ArrayStrategy.BY_KEY,
        params={"key": "id"},
    )
    assert ArrayF1().compute(node) == 1.0
    assert ArrayAccuracy().compute(node) == 1.0


def test_by_key_missing_element(tree_factory):
    item = ObjectFieldConfig(fields={"id": FieldConfig()})
    node = _array_node(
        tree_factory,
        [{"id": "a"}],
        [{"id": "a"}, {"id": "b"}],
        item_cfg=item,
        strategy=ArrayStrategy.BY_KEY,
        params={"key": "id"},
    )
    assert ArrayRecall().compute(node) == pytest.approx(0.5)
