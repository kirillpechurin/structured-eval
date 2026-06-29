"""Presence — 1.0 if the field exists in the actual document, else 0.0.

Unlike the comparison metrics, Presence overrides ``compute(node)`` (it inspects
the node, not a value pair), so it is exercised through a ScalarNode.
"""

from collections.abc import Callable

import pytest

from structured_eval import EvalContext, Presence
from structured_eval.model.nodes.scalar import ScalarNode

pytestmark = pytest.mark.unit


def test_present(context_factory: Callable[..., EvalContext]) -> None:
    ctx = context_factory({"x": "value"}, {"x": "value"})
    assert Presence().compute(ScalarNode(path="x", context=ctx)) == 1.0


def test_absent(context_factory: Callable[..., EvalContext]) -> None:
    ctx = context_factory({}, {"x": 1})
    assert Presence().compute(ScalarNode(path="x", context=ctx)) == 0.0


def test_ignores_expected_value(context_factory: Callable[..., EvalContext]) -> None:
    # present in actual even though it mismatches expected → still 1.0
    ctx = context_factory({"x": "a"}, {"x": "b"})
    assert Presence().compute(ScalarNode(path="x", context=ctx)) == 1.0
