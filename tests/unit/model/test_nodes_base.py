"""EvalNode tree base: lazy navigation, accessors, traversal.

Nodes never copy data — ``actual``/``expected`` resolve by navigating the shared
context. ``navigate`` distinguishes 'absent' (MISSING → surfaced as None).
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval import EvalContext
from structured_eval.model.nodes.base import MISSING, EvalNode, navigate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("obj", "path", "expected"),
    [
        ({"a": {"b": 1}}, "a.b", 1),  # dot path
        ({"items": [10, 20]}, "items[1]", 20),  # bracket index
        ({"items": [10, 20]}, "items[-1]", 20),  # negative index
    ],
    ids=["dot", "bracket", "negative"],
)
def test_navigate_resolves(obj: Any, path: Any, expected: Any) -> None:
    assert navigate(obj, path) == expected


def test_navigate_root_returns_same_object() -> None:
    obj = {"a": 1}
    assert navigate(obj, "$") is obj


@pytest.mark.parametrize(
    ("obj", "path"),
    [
        ({"a": 1}, "b"),  # missing key
        ({"items": [1]}, "items[5]"),  # out of range
        ({"items": [1]}, "items[x]"),  # non-int index
        ({"a": 1}, "a[0]"),  # index into non-list
    ],
    ids=["missing-key", "out-of-range", "non-int-index", "index-non-list"],
)
def test_navigate_absent_is_missing(obj: Any, path: Any) -> None:
    assert navigate(obj, path) is MISSING


def test_accessors_read_actual_and_expected(
    context_factory: Callable[..., EvalContext],
) -> None:
    ctx = context_factory({"a": 1}, {"a": 2})
    node = EvalNode(path="a", context=ctx)
    assert node.actual == 1
    assert node.expected == 2


def test_missing_actual_surfaces_as_none(
    context_factory: Callable[..., EvalContext],
) -> None:
    ctx = context_factory({"a": 1}, {})
    assert EvalNode(path="b", context=ctx).actual is None


def test_expected_none_when_no_expected(
    context_factory: Callable[..., EvalContext],
) -> None:
    ctx = context_factory({"a": 1}, None)
    assert EvalNode(path="a", context=ctx).expected is None


def test_diverging_expected_path(context_factory: Callable[..., EvalContext]) -> None:
    # array reordering: actual[0] ↔ expected[1]
    ctx = context_factory({"x": [1, 2]}, {"x": [2, 1]})
    node = EvalNode(path="x[0]", context=ctx, expected_path="x[1]")
    assert node.actual == 1
    assert node.expected == 1


def test_walk_visits_every_node(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
    paths = sorted(n.path for n in root.walk())
    assert "$" in paths and "a" in paths and "b.c" in paths


def test_leaves_are_only_scalars(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
    assert sorted(n.path for n in root.leaves()) == ["a", "b.c"]


def test_root_is_not_a_leaf(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1}, {"a": 1})
    assert not root.is_leaf()
