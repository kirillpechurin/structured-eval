"""Metric base classes, the name registry, and the per-node-type hierarchy.

`BaseMetric` is the registry root, and `Metric` adds the interface every metric
shares. The classes below them declare which node type a metric applies to,
which is what `TreeBuilder` cascades on:

- `FieldMetric` — one scalar leaf;
- `ObjectMetric` — one object node, root or nested;
- `ArrayMetric` — one array node;
- `RootMetric` — the document root only;
- `AnyNodeMetric` — every node, uniformly;
- `GenericMetric` — several node types, dispatched per kind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_eval.models.metrics import MetricResult
from structured_eval.models.nodes.array_node import ArrayNode
from structured_eval.models.nodes.base import EvalNode
from structured_eval.models.nodes.object_node import ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode

# What a metric's `compute` may return;
# `MetricRunner._apply` normalizes any of these to a `MetricResult`.
#
# A bare value / dict of sub-scores, optionally paired with structured
# `extra` via a tuple, or a ready `MetricResult`.
#
# `None` means the metric opted out of scoring this node.
MetricOutput = (
    float
    | dict[str, float]
    | tuple[float | dict[str, float], dict[str, Any]]
    | MetricResult
    | None
)

# Name → metric class. Populated automatically as BaseMetric subclasses are
# declared.
_METRIC_REGISTRY: dict[str, type] = {}


class BaseMetric(ABC):  # noqa: B024 — registry root; subclasses define the interface
    """Registry root for every metric — no evaluation interface of its own.

    - `name` is the key a scalar result lands under in `report.metrics` and
      `FieldScore.metrics`. A metric returning a `dict` writes its keys
      directly, and `name` is then only a registry handle;
    - declaring a subclass with a `name` registers it, which is what lets a
      config name the metric as a string;
    - a metric scores the node it was given and nothing else. One grading a
      whole subtree still reports a single value, with the per-field detail on
      that value's `extra`.

    Example:
        >>> from structured_eval.metrics import BaseMetric, resolve_metric
        >>> class Tally(BaseMetric):
        ...     name = "tally"
        >>> resolve_metric("tally").name      # declaring it registered the name
        'tally'
        >>> Tally(name="strict").name         # overridden for this instance only
        'strict'
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        """Bind an optional per-instance name.

        Args:
            name: Overrides the report key for this instance only, so two
                configurations of one metric can share a node instead of
                overwriting each other. The class registry is untouched.

        Raises:
            ValueError: If `name` is given but empty.
        """
        if name is not None:
            if not name:
                raise ValueError("metric name must be a non-empty string")
            self.name = name

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the subclass under its `name`, if it declares one."""
        super().__init_subclass__(**kwargs)
        if n := getattr(cls, "name", None):
            _METRIC_REGISTRY[n] = cls


class Metric[NodeT: EvalNode](BaseMetric):
    """The unified metric interface: `compute(node)` and `score(actual, expected)`.

    - `score` is the pure value-level comparison array alignment reuses;
    - `compute` is the node-level entry point, delegating to `score` by default,
      so a leaf comparison need only implement `score`;
    - aggregating metrics override `compute` and leave `score` at its default.

    `NodeT` pins the node type a subtype operates on: `ScalarNode` for fields,
    `ObjectNode` for objects, and so on.

    Example:
        >>> from typing import Any
        >>> from structured_eval.metrics import Metric
        >>> from structured_eval.models import ScalarNode
        >>> class Truthy(Metric[ScalarNode]):
        ...     name = "truthy"
        ...     def score(self, actual: Any, expected: Any) -> float:
        ...         return 1.0 if bool(actual) == bool(expected) else 0.0
        >>> Truthy().score("something", "other")
        1.0
    """

    def compute(self, node: NodeT) -> MetricOutput:
        """Grade one node, by comparing its own `actual` and `expected`."""
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float | dict[str, float]:
        """Compare two values; a `dict` reports several sub-scores at once."""
        raise NotImplementedError


class FieldMetric(Metric[ScalarNode]):
    """A leaf comparison applied to each `ScalarNode`.

    Implement `score` and inherit `compute`; a metric needing node context
    overrides `compute` instead. It is also the marker the engine dispatches on
    for scalars.

    Example:
        >>> from typing import Any
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import FieldMetric
        >>> from structured_eval.models import EvalConfig
        >>> class StartsWith(FieldMetric):
        ...     name = "starts_with"
        ...     def score(self, actual: Any, expected: Any) -> float:
        ...         return 1.0 if str(actual).startswith(str(expected)) else 0.0
        >>> StartsWith().score("INV-2024-001", "INV-")
        1.0
        >>> report = evaluate({"no": "INV-2024-001"}, {"no": "INV-"},
        ...                   EvalConfig(metrics=[StartsWith()]))
        >>> float(report.field_scores["no"].metrics["starts_with"])
        1.0
    """


class ObjectMetric(Metric[ObjectNode]):
    """Applies to each `ObjectNode`, root and nested alike.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ObjectMetric
        >>> from structured_eval.models import EvalConfig, ObjectNode
        >>> class ChildCount(ObjectMetric):
        ...     name = "child_count"
        ...     def compute(self, node: ObjectNode) -> float:
        ...         return float(len(node.children))
        >>> report = evaluate({"a": 1, "b": 2}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[ChildCount()]))
        >>> float(report.metrics["child_count"].representative())
        2.0
    """

    @abstractmethod
    def compute(self, node: ObjectNode) -> MetricOutput:
        """Grade one object node."""
        ...


class ArrayMetric(Metric[ArrayNode]):
    """Applies to each `ArrayNode`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ArrayMetric
        >>> from structured_eval.models import ArrayNode, EvalConfig
        >>> class ItemCount(ArrayMetric):
        ...     name = "item_count"
        ...     def compute(self, node: ArrayNode) -> float:
        ...         return float(len(node.items))
        >>> report = evaluate({"xs": [1, 2, 3]}, {"xs": [1, 2, 3]},
        ...                   EvalConfig(metrics=[ItemCount()]))
        >>> float(report.metrics["item_count"].representative())
        3.0
    """

    @abstractmethod
    def compute(self, node: ArrayNode) -> MetricOutput:
        """Grade one array node."""
        ...


class RootMetric(Metric[EvalNode]):
    """Applies only to the root node (path `"$"`); receives any `EvalNode`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import RootMetric
        >>> from structured_eval.models import EvalConfig, EvalNode
        >>> class LeafCount(RootMetric):
        ...     name = "leaf_count"
        ...     def compute(self, node: EvalNode) -> float:
        ...         return float(sum(1 for _ in node.leaves()))
        >>> report = evaluate({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}},
        ...                   EvalConfig(metrics=[LeafCount()]))
        >>> float(report.metrics["leaf_count"].representative())
        2.0
    """

    @abstractmethod
    def compute(self, node: EvalNode) -> MetricOutput:
        """Grade the document root."""
        ...


class AnyNodeMetric(Metric[EvalNode]):
    """Applies uniformly to *every* node — one `compute`, whatever the kind.

    The node-agnostic branch of the hierarchy, distinct from its two neighbours:

    - unlike `FieldMetric` / `ObjectMetric` / `ArrayMetric`, not pinned to one
      node type, and unlike `GenericMetric`, it does not dispatch per kind;
    - unlike `RootMetric`, which is also `Metric[EvalNode]`, it is admitted
      everywhere rather than only at the root.

    `MeanScore`, the default representative, lives here.
    """

    @abstractmethod
    def compute(self, node: EvalNode) -> MetricOutput:
        """Grade any node, whatever its kind."""
        ...


class GenericMetric(BaseMetric):
    """Metrics spanning several node types, outside the single-`compute` shape.

    Override whichever per-kind methods apply:

    - `compute_<kind>` for node mode;
    - `score_<kind>` for value mode, which array alignment reuses.

    `MetricInvoker` dispatches by kind. Unlike `Metric`, this branch has no
    default `compute` delegating to `score`, so a metric without the node's
    `compute_<kind>` has nothing to grade it with — which is why `TreeBuilder`
    admits it onto a node only where that method exists.

    Example:
        >>> from typing import Any
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import GenericMetric
        >>> from structured_eval.models import EvalConfig, ObjectNode
        >>> class Width(GenericMetric):
        ...     name = "width"
        ...     def compute_object(self, node: ObjectNode) -> float:
        ...         return float(len(node.children))
        ...     def score_object(self, actual: Any, expected: Any) -> float:
        ...         return float(len(actual))
        >>> report = evaluate({"a": 1, "b": 2}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[Width()]))
        >>> float(report.metrics["width"].representative())
        2.0
    """


def get_metric_class(name: str) -> type:
    """Resolve a metric class by its registered `name`.

    Args:
        name: A registered metric name, e.g. `"object_f1"`.

    Returns:
        The class registered under that name.

    Raises:
        KeyError: If no metric is registered under `name`.

    Example:
        >>> from structured_eval.metrics import get_metric_class
        >>> get_metric_class("object_f1").__name__
        'ObjectF1'
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Known: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def resolve_metric(spec: str | BaseMetric) -> BaseMetric:
    """Coerce a metric spec to a `BaseMetric` instance.

    The single resolver shared by the engine, array alignment and the
    match-criterion helper. `None` is *not* handled here — callers supply their
    own default.

    Args:
        spec: A metric instance, or a registered name to instantiate with no
            arguments.

    Returns:
        The instance itself, or a fresh one built from the name.

    Example:
        >>> from structured_eval.metrics import Numeric, resolve_metric
        >>> resolve_metric("numeric").name
        'numeric'
        >>> resolve_metric(Numeric(tolerance=0.05)).tolerance
        0.05
    """
    if isinstance(spec, str):
        instance = get_metric_class(spec)()
        assert isinstance(instance, BaseMetric)
        return instance
    assert isinstance(spec, BaseMetric)
    return spec
