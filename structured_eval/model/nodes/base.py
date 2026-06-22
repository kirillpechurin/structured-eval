from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.model.context import EvalContext
from structured_eval.model.metric_result import MetricResult
from structured_eval.utils.paths import MISSING, navigate

# Re-exported for back-compat: ``navigate`` / ``MISSING`` now live in
# ``structured_eval.utils.paths`` (a lower layer with no model dependency).
__all__ = ["MISSING", "EvalNode", "navigate"]


class EvalNode(BaseModel):
    """A node in the evaluation tree.

    Holds its ``path`` and a shared reference to the ``EvalContext``; data is
    never copied — ``actual``/``expected`` are resolved lazily by navigating the
    context's documents. ``expected_path`` defaults to ``path``; it diverges
    only for array items aligned out of order (``expected[1]`` ↔ ``actual[0]``),
    so each side navigates its own index. ``metric_results`` accumulates each
    requested metric's value at this node (filled by the engine in phase 2).

    ``key_metric`` is the node's *representative* metric — the single score that
    bubbles up to a parent's aggregation (and, at the root, to ``report.score``).
    It is computed last (its logic may depend on the node's other metrics) and
    defaults to ``MeanScore`` (the arithmetic mean of the node's own metrics).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    context: EvalContext
    expected_path: str | None = None
    weight: float = 1.0  # relative importance for weighted aggregation (OverallLeafScore, object metrics)
    metrics: list[Any] = Field(default_factory=list)  # list[BaseMetric] resolved for this node
    key_metric: Any = None  # BaseMetric: this node's representative score (parents read it)
    threshold: float = 1.0  # bar the representative score must clear to count as a TP
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)

    @property
    def actual(self) -> Any:
        value = navigate(self.context.actual, self.path)
        return None if value is MISSING else value

    @property
    def expected(self) -> Any:
        if self.context.expected is None:
            return None
        value = navigate(self.context.expected, self.expected_path or self.path)
        return None if value is MISSING else value

    @property
    def representative(self) -> float:
        """The node's single representative score: its ``key_metric``'s value.

        Every node always carries a ``key_metric`` (the engine defaults it to
        ``MeanScore``) and at least one metric for it to summarise, so by the
        time anyone reads this the value exists. A parent reads its already
        computed children's representatives to aggregate (post-order); the root's
        is ``report.score``. Missing is a programming error, not a fallback.
        """
        km = self.key_metric
        if km is None:
            raise ValueError(f"node {self.path!r} has no key_metric")
        value = self.metric_results.get(km.name)
        if value is None:
            raise ValueError(f"node {self.path!r}: key_metric {km.name!r} has no computed value")
        return float(value)

    # ── traversal ──────────────────────────────────────────────────────────
    # Children are discovered by duck-typing (``children`` on objects, ``items``
    # on arrays) so the base node need not import its own subclasses.

    def children_nodes(self) -> Iterator[EvalNode]:
        """Yield the node's direct child nodes (none for a scalar leaf)."""
        children = getattr(self, "children", None)
        if isinstance(children, dict):
            yield from children.values()
        items = getattr(self, "items", None)
        if isinstance(items, list):
            yield from items

    def is_leaf(self) -> bool:
        """True for a scalar node (no object children, no array items)."""
        return getattr(self, "children", None) is None and getattr(self, "items", None) is None

    def walk(self) -> Iterator[EvalNode]:
        """Depth-first traversal yielding this node and every descendant."""
        yield self
        for child in self.children_nodes():
            yield from child.walk()

    def leaves(self) -> Iterator[EvalNode]:
        """Yield every scalar (leaf) node at or beneath this node."""
        for node in self.walk():
            if node.is_leaf():
                yield node
