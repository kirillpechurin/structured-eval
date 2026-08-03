"""object_utils — the verdicts an object metric aggregates over its children.

The ``Object*`` metric tests cover the arithmetic; this file covers the helper's
own contract, which the metrics only partly reach: how a bar is chosen for each
matched child, and how a child's name is read off its path.
"""

from typing import Any

import pytest

from structured_eval.metrics.utils import object_utils as obj
from structured_eval.metrics.utils.calculate import WeightMode
from structured_eval.models import EvalConfig, FieldConfig, ObjectNode
from tests.conftest import build_tree

pytestmark = pytest.mark.unit

CONFIG = EvalConfig(
    fields={"a": FieldConfig(threshold=0.5, weight=3.0), "b": FieldConfig()}
)


def _object() -> ObjectNode:
    root = build_tree({"a": 1, "b": 2}, {"a": 1, "b": 2}, CONFIG)
    assert isinstance(root, ObjectNode)
    return root


@pytest.mark.parametrize(
    ("thresholds", "bars"),
    [
        (None, [0.5, 1.0]),
        (0.8, [0.8, 0.8]),
        ({"a": 0.9}, [0.9, 1.0]),
    ],
    ids=["own", "single-bar", "per-field"],
)
def test_a_bar_is_chosen_per_child(thresholds: Any, bars: list[float]) -> None:
    # No override → each child keeps the bar it was configured with; a single
    # float replaces every one of them; a dict names the children it overrides
    # and leaves the rest on their own bar.
    verdicts = obj.matched_verdicts(_object(), thresholds=thresholds)
    assert [threshold for _, threshold, _ in verdicts] == bars


@pytest.mark.parametrize(
    ("weight_mode", "weights"),
    [(WeightMode.PROPORTIONAL, [3.0, 1.0]), (WeightMode.NONE, [1.0, 1.0])],
    ids=["proportional", "none"],
)
def test_weights_follow_the_weight_mode(
    weight_mode: WeightMode, weights: list[float]
) -> None:
    verdicts = obj.matched_verdicts(_object(), weight_mode=weight_mode)
    assert [weight for _, _, weight in verdicts] == weights


@pytest.mark.parametrize(
    ("path", "name"),
    [("total", "total"), ("a.b", "b"), ("a.b[0]", "b"), ("lines[2]", "lines")],
    ids=["bare", "nested", "indexed-nested", "indexed"],
)
def test_a_child_is_named_by_its_last_segment(path: str, name: str) -> None:
    # `score_policy` and per-field thresholds are keyed by name, not by path —
    # the same policy has to apply to every element of an array.
    assert obj.leaf_name(path) == name
