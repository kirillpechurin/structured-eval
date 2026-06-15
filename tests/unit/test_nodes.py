"""Unit tests for the EvalNode tree: lazy navigation, traversal, leaves.

Nodes never copy data — ``actual``/``expected`` resolve by navigating the shared
context. ``_navigate`` distinguishes 'absent' (MISSING → surfaced as None).
"""

from __future__ import annotations

import pytest

from structured_eval.model.nodes.base import MISSING, EvalNode, _navigate

pytestmark = pytest.mark.unit


class TestNavigate:
    def test_root(self):
        obj = {"a": 1}
        assert _navigate(obj, "$") is obj

    def test_dot_path(self):
        assert _navigate({"a": {"b": 1}}, "a.b") == 1

    def test_bracket_index(self):
        assert _navigate({"items": [10, 20]}, "items[1]") == 20

    def test_negative_index(self):
        assert _navigate({"items": [10, 20]}, "items[-1]") == 20

    def test_missing_key(self):
        assert _navigate({"a": 1}, "b") is MISSING

    def test_out_of_range(self):
        assert _navigate({"items": [1]}, "items[5]") is MISSING

    def test_non_int_index(self):
        assert _navigate({"items": [1]}, "items[x]") is MISSING

    def test_index_into_non_list(self):
        assert _navigate({"a": 1}, "a[0]") is MISSING


class TestNodeAccessors:
    def test_actual_and_expected(self, context_factory):
        ctx = context_factory({"a": 1}, {"a": 2})
        node = EvalNode(path="a", context=ctx)
        assert node.actual == 1
        assert node.expected == 2

    def test_missing_surfaces_as_none(self, context_factory):
        ctx = context_factory({"a": 1}, {})
        node = EvalNode(path="b", context=ctx)
        assert node.actual is None

    def test_expected_none_when_no_expected(self, context_factory):
        ctx = context_factory({"a": 1}, None)
        node = EvalNode(path="a", context=ctx)
        assert node.expected is None

    def test_diverging_expected_path(self, context_factory):
        # array reordering: actual[0] ↔ expected[1]
        ctx = context_factory({"x": [1, 2]}, {"x": [2, 1]})
        node = EvalNode(path="x[0]", context=ctx, expected_path="x[1]")
        assert node.actual == 1
        assert node.expected == 1


class TestTraversal:
    def test_leaf_and_walk(self, tree_factory):
        root = tree_factory({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
        paths = sorted(n.path for n in root.walk())
        assert "$" in paths and "a" in paths and "b.c" in paths

    def test_leaves_only_scalars(self, tree_factory):
        root = tree_factory({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
        leaf_paths = sorted(n.path for n in root.leaves())
        assert leaf_paths == ["a", "b.c"]

    def test_root_is_not_leaf(self, tree_factory):
        root = tree_factory({"a": 1}, {"a": 1})
        assert not root.is_leaf()
