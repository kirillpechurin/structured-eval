"""Unit tests for object metrics, run over a TreeBuilder-built ObjectNode root.

Object metrics grade only the *scalar* fields of an object: matched fields are
both predicted and expected, ``missing`` keys are FN, ``spurious`` keys are FP
(only when ExtraKeysPolicy.PENALIZE). Nested children grade at their own node.
"""

from collections.abc import Callable

import pytest

from structured_eval import (
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectAccuracy,
    ObjectF1,
    ObjectPrecision,
    ObjectPRF1,
    ObjectRecall,
    ObjectTypeValidity,
)
from structured_eval.metrics.utils.calculate import GradingMode, WeightMode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode

pytestmark = pytest.mark.unit


def test_f1_perfect(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert isinstance(root, ObjectNode)
    assert ObjectF1().compute(root) == 1.0


def test_f1_one_wrong(tree_factory: Callable[..., EvalNode]) -> None:
    # both fields present (TP/FP by value), one wrong → tp=1, predicted=2, expected=2
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2})
    assert isinstance(root, ObjectNode)
    assert ObjectF1().compute(root) == pytest.approx(0.5)


def test_precision_recall_with_missing_and_extra(
    tree_factory: Callable[..., EvalNode],
) -> None:
    # actual has a,c ; expected has a,b. With PENALIZE: missing=[b], spurious=[c]
    cfg = EvalConfig(extra_keys=ExtraKeysPolicy.PENALIZE)
    root = tree_factory({"a": 1, "c": 3}, {"a": 1, "b": 2}, cfg)
    assert isinstance(root, ObjectNode)
    # tp=1 (a), predicted = 1 matched + 1 spurious = 2, expected = 1 matched + 1 missing = 2
    assert ObjectPrecision().compute(root) == pytest.approx(0.5)
    assert ObjectRecall().compute(root) == pytest.approx(0.5)


def test_extra_keys_ignored_by_default(tree_factory: Callable[..., EvalNode]) -> None:
    # default IGNORE: spurious empty → extra key does not hurt precision
    root = tree_factory({"a": 1, "c": 3}, {"a": 1})
    assert isinstance(root, ObjectNode)
    assert ObjectPrecision().compute(root) == 1.0


def test_prf1_returns_dict(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2})
    assert isinstance(root, ObjectNode)
    result = ObjectPRF1().compute(root)
    assert set(result) == {"object_precision", "object_recall", "object_f1"}
    assert result["object_f1"] == pytest.approx(0.5)


def test_accuracy_is_soft(tree_factory: Callable[..., EvalNode]) -> None:
    # accuracy rewards partial: token_f1 as criterion on a near-miss
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert isinstance(root, ObjectNode)
    assert ObjectAccuracy().compute(root) == 1.0


def test_accuracy_missing_counts_zero(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1}, {"a": 1, "b": 2})
    assert isinstance(root, ObjectNode)
    # one matched (1.0) + one missing (0) over denom 2
    assert ObjectAccuracy().compute(root) == pytest.approx(0.5)


def test_soft_mode_fractional(tree_factory: Callable[..., EvalNode]) -> None:
    from structured_eval import TokenF1

    cfg = EvalConfig(metrics=[TokenF1()])  # field metric cascades to every scalar
    root = tree_factory({"name": "the quick brown fox"}, {"name": "the quick fox"}, cfg)
    assert isinstance(root, ObjectNode)
    soft = ObjectF1(score_policy={"name": "token_f1"}, mode=GradingMode.SOFT).compute(
        root
    )
    assert 0.0 < soft < 1.0


def test_validity_type_check(tree_factory: Callable[..., EvalNode]) -> None:
    # "100" (str) vs 100 (number): present but wrong type → 0
    root = tree_factory({"a": "100", "b": 2}, {"a": 100, "b": 2})
    assert isinstance(root, ObjectNode)
    assert ObjectTypeValidity().compute(root) == pytest.approx(0.5)


def test_validity_checks_container_types(tree_factory: Callable[..., EvalNode]) -> None:
    # all child kinds count: a list where an object is expected is type-invalid
    root = tree_factory(
        {"price": 199, "instructor": ["X"], "tags": ["a"]},
        {"price": 199, "instructor": {"name": "X"}, "tags": ["a", "b"]},
    )
    assert isinstance(root, ObjectNode)
    # price: number✓, instructor: array vs object ✗, tags: array✓ → 2/3
    assert ObjectTypeValidity().compute(root) == pytest.approx(2 / 3)


def test_validity_vacuous_when_no_scalars(
    tree_factory: Callable[..., EvalNode],
) -> None:
    root = tree_factory({}, {})
    assert isinstance(root, ObjectNode)
    assert ObjectTypeValidity().compute(root) == 1.0


def test_empty_object_vacuously_perfect(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({}, {})
    assert isinstance(root, ObjectNode)
    assert ObjectF1().compute(root) == 1.0
    assert ObjectAccuracy().compute(root) == 1.0


# ── weighting (weight_mode) ─────────────────────────────────────────────────


def _weighted_tree(tree_factory: Callable[..., EvalNode]) -> ObjectNode:
    from structured_eval import FieldConfig

    # a correct, b wrong; b is 3× as important as a.
    cfg = EvalConfig(
        fields={"a": FieldConfig(weight=1.0), "b": FieldConfig(weight=3.0)}
    )
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2}, cfg)
    assert isinstance(root, ObjectNode)
    return root


def test_f1_weighted_by_default(tree_factory: Callable[..., EvalNode]) -> None:
    root = _weighted_tree(tree_factory)
    # tp=1 (a, w1), predicted=expected=4 (w1+w3) → 0.25 (vs 0.5 unweighted)
    assert ObjectF1().compute(root) == pytest.approx(0.25)


def test_f1_weight_mode_none_restores_plain(
    tree_factory: Callable[..., EvalNode],
) -> None:
    root = _weighted_tree(tree_factory)
    assert ObjectF1(weight_mode=WeightMode.NONE).compute(root) == pytest.approx(0.5)


def test_accuracy_weighted_by_default(tree_factory: Callable[..., EvalNode]) -> None:
    root = _weighted_tree(tree_factory)
    # numer = 1*1 + 3*0 = 1, denom = 1+3 = 4 → 0.25 (vs 0.5 unweighted)
    assert ObjectAccuracy().compute(root) == pytest.approx(0.25)
    assert ObjectAccuracy(weight_mode=WeightMode.NONE).compute(root) == pytest.approx(
        0.5
    )


def test_weighted_missing_field(tree_factory: Callable[..., EvalNode]) -> None:
    # a present+correct (w1), b missing (w3): weighted recall/accuracy drop more.
    cfg = EvalConfig(
        fields={"a": FieldConfig(weight=1.0), "b": FieldConfig(weight=3.0)}
    )
    root = tree_factory({"a": 1}, {"a": 1, "b": 2}, cfg)
    assert isinstance(root, ObjectNode)
    # accuracy: numer=1, denom=1(matched)+3(missing)=4 → 0.25 (vs 0.5 unweighted)
    assert ObjectAccuracy().compute(root) == pytest.approx(0.25)
    assert ObjectRecall().compute(root) == pytest.approx(0.25)
