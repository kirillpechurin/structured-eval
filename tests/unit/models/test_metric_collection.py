"""MetricCollection — one named metric's values across the whole tree.

``report.metrics[name]``. The reductions answer different questions about the
same map: ``mean``/``min``/``max`` summarise every node that produced the
metric, ``root`` is the document-level value, and ``representative`` picks
whichever of the two is meaningful. An empty collection answers ``0.0`` rather
than raising — a metric that ran nowhere is not an error.
"""

from typing import Any

import pytest

from structured_eval.models import MetricCollection, MetricResult

pytestmark = pytest.mark.unit


def _coll(**by_path: float) -> MetricCollection:
    return MetricCollection(
        name="m", by_path={p: MetricResult(v) for p, v in by_path.items()}
    )


def test_values_are_the_nodes_results() -> None:
    assert _coll(a=1.0, b=0.0).values() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [("mean", 0.5), ("min", 0.0), ("max", 1.0)],
    ids=["mean", "min", "max"],
)
def test_reductions_over_the_tree(reduce: str, expected: float) -> None:
    assert getattr(_coll(a=1.0, b=0.0), reduce)() == pytest.approx(expected)


@pytest.mark.parametrize("reduce", ["mean", "min", "max"])
def test_an_empty_collection_reduces_to_zero(reduce: str) -> None:
    # A metric that ran on no node has no value to report; that is not an error.
    assert getattr(_coll(), reduce)() == 0.0


@pytest.mark.parametrize(
    ("by_path", "root", "representative"),
    [
        ({"$": 0.25, "a": 1.0}, 0.25, 0.25),
        ({"a": 1.0, "b": 0.0}, None, 0.5),
        ({}, None, 0.0),
    ],
    ids=["scored-at-root", "fields-only", "ran-nowhere"],
)
def test_root_and_representative(
    by_path: dict[str, float], root: float | None, representative: float
) -> None:
    # `representative` answers "how did the document score" when the document
    # itself was scored, and falls back to "how did the fields average" when it
    # was not — the two questions, one table.
    coll = _coll(**by_path)
    assert coll.root() == root
    assert coll.representative() == pytest.approx(representative)


def test_extra_collects_only_the_non_empty_payloads() -> None:
    coll = MetricCollection(
        name="m",
        by_path={
            "a": MetricResult(1.0),
            "b": MetricResult(0.0, {"reason": "invented"}),
        },
    )
    assert coll.extra == [{"reason": "invented"}]


def test_extra_values_flattens_a_list_valued_key() -> None:
    coll = MetricCollection(
        name="field_faithfulness",
        by_path={
            "a": MetricResult(0.0, {"hallucinated": ["x"]}),
            "b": MetricResult(0.0, {"hallucinated": ["y", "z"]}),
            "c": MetricResult(1.0),
        },
    )
    assert coll.extra_values("hallucinated") == ["x", "y", "z"]


def test_extra_values_of_an_absent_key_is_empty() -> None:
    values: list[Any] = _coll(a=1.0).extra_values("nope")
    assert values == []
