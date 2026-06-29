"""The single way to run a metric, whatever input is available.

Every metric is invoked through ``MetricInvoker`` — never by calling ``compute``
/ ``compute_<kind>`` / ``score`` directly. Two input modes:

* ``on_node`` — a node is available: grade it. A ``Metric`` uses ``compute``; a
  ``GenericMetric`` dispatches to the ``compute_<kind>`` for the node's type.
* ``on_values`` — only raw ``actual`` / ``expected`` (array alignment, before any
  node exists): compare them. A ``Metric`` uses ``score``; a ``GenericMetric``
  dispatches to the ``score_<kind>`` for the kind inferred from the value's shape.

Each mode has a ``scalar_*`` variant that narrows the result to a single
``float`` (rejecting a dict of sub-scores) — that narrowing is the caller's
contract, hence its own method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import BaseMetric, GenericMetric, Metric, MetricOutput
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode

if TYPE_CHECKING:
    from structured_eval.model.nodes.base import EvalNode

# A GenericMetric's per-kind method names, by node class, for each input mode.
GENERIC_NODE_METHOD: dict[type, str] = {
    ScalarNode: "compute_scalar",
    ObjectNode: "compute_object",
    ArrayNode: "compute_array",
}
GENERIC_SCORE_METHOD: dict[type, str] = {
    ScalarNode: "score_scalar",
    ObjectNode: "score_object",
    ArrayNode: "score_array",
}


def _kind_of(actual: Any, expected: Any) -> type:
    """The node class a raw value pair would build (mirrors ``TreeBuilder``)."""
    ref = expected if expected is not None else actual
    if isinstance(ref, dict):
        return ObjectNode
    if isinstance(ref, list):
        return ArrayNode
    return ScalarNode


class MetricInvoker:
    """Runs ``self.metric`` in either input mode; see module docstring."""

    def __init__(self, metric: BaseMetric):
        self.metric = metric

    def on_node(self, node: EvalNode) -> MetricOutput:
        metric = self.metric
        if isinstance(metric, GenericMetric):
            return self._dispatch_generic(GENERIC_NODE_METHOD.get(type(node)), node)
        assert isinstance(metric, Metric)  # every non-generic metric has compute(node)
        return metric.compute(node)

    def on_values(self, actual: Any, expected: Any) -> MetricOutput:
        metric = self.metric
        if isinstance(metric, GenericMetric):
            method = GENERIC_SCORE_METHOD.get(_kind_of(actual, expected))
            return self._dispatch_generic(method, actual, expected)
        assert isinstance(metric, Metric)  # value comparison needs score()
        return metric.score(actual, expected)

    def scalar_on_node(self, node: EvalNode) -> float:
        return self._scalar(self.on_node(node), node.path)

    def scalar_on_values(self, actual: Any, expected: Any) -> float:
        return self._scalar(self.on_values(actual, expected), "<values>")

    def _dispatch_generic(self, method: str | None, *args: Any) -> MetricOutput:
        if method is None or not hasattr(self.metric, method):
            return None
        result: MetricOutput = getattr(self.metric, method)(*args)
        return result

    def _scalar(self, result: Any, where: str) -> float:
        assert isinstance(result, (int, float)), (
            f"metric {self.metric.name!r} must yield a scalar at {where}, "
            f"got {type(result).__name__}"
        )
        return float(result)
