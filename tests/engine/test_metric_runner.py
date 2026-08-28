"""MetricRunner — phase 2, computing every node's metrics in place.

The full pipeline is covered by `test_evaluator.py`; what is pinned here is
the runner's own contract, driven by hand-built nodes so the order is
observable: post-order across the tree, `key_metric` last within a node, and
a metric returning `None` skipped rather than scored.
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval.engine.metric_runner import MetricRunner
from structured_eval.metrics.base import AnyNodeMetric
from structured_eval.models import EvalConfig, EvalContext, EvalNode, ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode

pytestmark = pytest.mark.engine


class Recorder(AnyNodeMetric):
    """Scores a constant and remembers, in order, where it was asked.

    `log` is shared between instances when several of them sit on one node,
    so the order *within* a node is observable as well as the order across the
    tree.
    """

    name = "recorder"

    def __init__(
        self,
        value: float | None = 1.0,
        log: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.value = value
        self.seen: list[str] = []
        self.log = log

    def compute(self, node: EvalNode) -> float | None:
        self.seen.append(node.path)
        if self.log is not None:
            self.log.append(self.name)
        return self.value


def test_a_node_without_a_key_metric_still_gets_its_metrics(
    context_factory: Callable[..., EvalContext],
) -> None:
    # `key_metric` is typed permissively and defaults to None; a node assembled
    # by hand may have none. The runner computes what the node does carry and
    # leaves it without a representative — `EvalNode.representative` is what
    # refuses to invent one, and only its readers care.
    metric = Recorder(0.25)
    node = ScalarNode(path="a", context=context_factory({"a": 1}), metrics=[metric])

    MetricRunner().run(node)

    assert node.metric_results["recorder"] == 0.25
    assert metric.seen == ["a"]


def test_children_are_computed_before_their_parent(
    context_factory: Callable[..., EvalContext],
) -> None:
    # Post-order, because an aggregating parent reads its children's already
    # computed representatives.
    metric = Recorder()
    context = context_factory({"vendor": {"name": "Acme"}})
    leaf = ScalarNode(path="vendor.name", context=context, metrics=[metric])
    inner = ObjectNode(
        path="vendor", context=context, metrics=[metric], children={"name": leaf}
    )
    root = ObjectNode(
        path="$", context=context, metrics=[metric], children={"vendor": inner}
    )

    MetricRunner().run(root)

    assert metric.seen == ["vendor.name", "vendor", "$"]


def test_the_key_metric_runs_last_within_a_node(
    context_factory: Callable[..., EvalContext],
) -> None:
    # Its logic may depend on the node's other metrics — the default MeanScore
    # averages them — so it cannot run before they have values. The key metric
    # sits in the middle of `metrics` here, to show that position does not
    # decide it.
    order: list[str] = []
    key = Recorder(log=order, name="key")
    node = ScalarNode(
        path="a",
        context=context_factory({"a": 1}),
        metrics=[
            Recorder(log=order, name="first"),
            key,
            Recorder(log=order, name="second"),
        ],
        key_metric=key,
    )

    MetricRunner().run(node)

    assert order == ["first", "second", "key"]


def test_a_key_metric_not_listed_among_the_metrics_still_runs(
    context_factory: Callable[..., EvalContext],
) -> None:
    # The usual shape: `TreeBuilder` resolves MeanScore fresh rather than
    # finding it on the node.
    listed, key = Recorder(name="listed"), Recorder(0.5, name="key")
    node = ScalarNode(
        path="a", context=context_factory({"a": 1}), metrics=[listed], key_metric=key
    )

    MetricRunner().run(node)

    assert node.metric_results["key"] == 0.5
    assert node.representative == 0.5


@pytest.mark.parametrize(
    ("returned", "expected"),
    [
        (0.25, {"recorder": 0.25}),
        (None, {}),
    ],
    ids=["a-score", "opted-out"],
)
def test_a_metric_returning_nothing_is_skipped(
    context_factory: Callable[..., EvalContext],
    returned: float | None,
    expected: dict[str, Any],
) -> None:
    # `None` means the metric had nothing to say here — a null leaf for
    # FieldFaithfulness, an unparseable value for UrlMatch. That is not a zero.
    node = ScalarNode(
        path="a", context=context_factory({"a": 1}), metrics=[Recorder(returned)]
    )

    MetricRunner().run(node)

    assert dict(node.metric_results) == expected


def test_the_runner_needs_no_configuration(
    context_factory: Callable[..., EvalContext],
) -> None:
    # Everything it needs was decided in phase 1 and travels on the node, so one
    # runner serves every document without carrying state between them.
    context = context_factory({"a": 1}, config=EvalConfig())
    runner = MetricRunner()
    for path in ("a", "b"):
        node = ScalarNode(path=path, context=context, metrics=[Recorder(1.0)])
        runner.run(node)
        assert node.metric_results["recorder"] == 1.0
