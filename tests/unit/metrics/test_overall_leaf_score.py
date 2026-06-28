"""OverallLeafScore — weighted mean over the scalar leaves of the whole tree.

Run on the built tree root, mirroring how the engine fires a root metric.
"""

import pytest

from structured_eval import EvalConfig, FieldConfig, OverallLeafScore, TokenF1

pytestmark = pytest.mark.unit


def test_perfect(tree_factory):
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert OverallLeafScore().compute(root) == 1.0


def test_half(tree_factory):
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2})
    assert OverallLeafScore().compute(root) == pytest.approx(0.5)


def test_weighted_by_field_weight(tree_factory):
    cfg = EvalConfig(fields={"a": FieldConfig(weight=3.0), "b": FieldConfig(weight=1.0)})
    # a correct (w3), b wrong (w1) → 3/(3+1)
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2}, cfg)
    assert OverallLeafScore().compute(root) == pytest.approx(0.75)


def test_uses_each_leaf_key_metric(tree_factory):
    cfg = EvalConfig(fields={"name": FieldConfig(metrics=[TokenF1()], key_metric=TokenF1())})
    root = tree_factory({"name": "the quick fox"}, {"name": "the quick brown fox"}, cfg)
    assert 0.0 < OverallLeafScore().compute(root) < 1.0


def test_empty_tree_is_vacuously_one(tree_factory):
    root = tree_factory({}, {})
    assert OverallLeafScore().compute(root) == 1.0
