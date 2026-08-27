"""The single way to run a metric, whatever input is available."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import BaseMetric, GenericMetric, Metric, MetricOutput
from structured_eval.models.nodes.array_node import ArrayNode
from structured_eval.models.nodes.object_node import ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode

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
    """The node class a raw value pair would build (mirrors `TreeBuilder`)."""
    ref = expected if expected is not None else actual
    if isinstance(ref, dict):
        return ObjectNode
    if isinstance(ref, list):
        return ArrayNode
    return ScalarNode


class MetricInvoker:
    """Runs one metric, in whichever of the two input modes the caller has.

    A metric is never called directly: the invoker is what knows how each shape
    of metric is entered. The two modes differ in what the caller holds:

    - node mode (`on_node`) — the engine has a built node, with its children,
      alignment and already-computed child scores;
    - value mode (`on_values`) — there is no node yet, only a pair of raw
      values, which is what array alignment scores candidate pairs with.

    A `GenericMetric` is dispatched by node kind in either mode.

    Attributes:
        metric: The bound metric; its `name` is the key a result lands under.
    """

    def __init__(self, metric: BaseMetric):
        """Bind the metric to invoke.

        Args:
            metric: Any metric — the mode it is entered in is the caller's
                choice, not the metric's.
        """
        self.metric = metric

    def on_node(self, node: EvalNode) -> MetricOutput:
        """Grade a node, dispatching by kind for a `GenericMetric`.

        Args:
            node: The node to grade.

        Returns:
            Whatever the metric produced, or `None` — either because the metric
            opted out of this node, or because it is a `GenericMetric` with no
            `compute_<kind>` for this kind.
        """
        metric = self.metric
        if isinstance(metric, GenericMetric):
            return self._dispatch_generic(GENERIC_NODE_METHOD.get(type(node)), node)
        assert isinstance(metric, Metric)  # every non-generic metric has compute(node)
        return metric.compute(node)

    def on_values(self, actual: Any, expected: Any) -> MetricOutput:
        """Compare two raw values, before any node exists.

        The kind a `GenericMetric` is dispatched on is read off the values the
        way `TreeBuilder` would read it off them.

        Args:
            actual: The produced value.
            expected: The value it is compared against.

        Returns:
            Whatever the metric produced, or `None` — either because the metric
            opted out of this pair, or because it is a `GenericMetric` with no
            `score_<kind>` for this kind.
        """
        metric = self.metric
        if isinstance(metric, GenericMetric):
            method = GENERIC_SCORE_METHOD.get(_kind_of(actual, expected))
            return self._dispatch_generic(method, actual, expected)
        assert isinstance(metric, Metric)  # value comparison needs score()
        return metric.score(actual, expected)

    def scalar_on_node(self, node: EvalNode) -> float:
        """`on_node`, narrowed to a single `float`.

        Args:
            node: The node to grade.

        Returns:
            The metric's score for it.

        Raises:
            AssertionError: If the metric yielded sub-scores or nothing at all,
                where the caller needs one number.
        """
        return self._scalar(self.on_node(node), node.path)

    def scalar_on_values(self, actual: Any, expected: Any) -> float:
        """`on_values`, narrowed to a single `float`.

        Args:
            actual: The produced value.
            expected: The value it is compared against.

        Returns:
            The metric's score for the pair.

        Raises:
            AssertionError: If the metric yielded sub-scores or nothing at all,
                where the caller needs one number.
        """
        return self._scalar(self.on_values(actual, expected), "<values>")

    def _dispatch_generic(self, method: str | None, *args: Any) -> MetricOutput:
        """Call the per-kind method, or opt out when it is not implemented."""
        if method is None or not hasattr(self.metric, method):
            return None
        result: MetricOutput = getattr(self.metric, method)(*args)
        return result

    def _scalar(self, result: Any, where: str) -> float:
        """Narrow a metric's output to one number, `where` naming the caller."""
        assert isinstance(result, (int, float)), (
            f"metric {self.metric.name!r} must yield a scalar at {where}, "
            f"got {type(result).__name__}"
        )
        return float(result)
