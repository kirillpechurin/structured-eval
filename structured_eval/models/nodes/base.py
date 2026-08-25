"""`EvalNode`, the base every tree node shares, and the `NodeType` tag."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.models.context import EvalContext  # noqa: TC001
from structured_eval.models.metrics import MetricResult  # noqa: TC001
from structured_eval.utils.paths import MISSING, navigate

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["EvalNode", "NodeType"]


class NodeType(StrEnum):
    """The kind of tree node a `FieldScore` describes.

    Attributes:
        SCALAR: A leaf value.
        OBJECT: A dict.
        ARRAY: A list.
    """

    SCALAR = "scalar"
    OBJECT = "object"
    ARRAY = "array"


class EvalNode(BaseModel):
    """A node in the evaluation tree.

    Data is never copied: a node holds its `path` and a shared reference to the
    `EvalContext`, and resolves `actual` / `expected` lazily by navigating the
    context's documents.

    Attributes:
        path: This node's dot-and-bracket path in the actual document.
        context: The sample data every node in the tree shares.
        expected_path: Where to look on the expected side; defaults to `path`
            and diverges only for array items aligned out of order
            (`expected[1]` ↔ `actual[0]`), so each side navigates its own index.
        weight: Relative importance in the parent's weighted aggregation.
        metrics: The metrics resolved for this node.
        key_metric: This node's *representative* metric — the single score that
            bubbles up to a parent's aggregation, and at the root to
            `report.score`. Computed last, since its logic may depend on the
            node's other metrics; defaults to `MeanScore`.
        threshold: The bar the representative score must clear to count as a TP.
        metric_results: Each requested metric's value here, filled by phase 2.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    context: EvalContext
    expected_path: str | None = None
    weight: float = 1.0
    metrics: list[Any] = Field(default_factory=list)
    key_metric: Any = None
    threshold: float = 1.0
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)

    @property
    def actual(self) -> Any:
        """This node's value in the actual document; `None` when absent."""
        value = navigate(self.context.actual, self.path)
        return None if value is MISSING else value

    @property
    def expected(self) -> Any:
        """This node's value in the expected document; `None` when absent."""
        if self.context.expected is None:
            return None
        value = navigate(self.context.expected, self.expected_path or self.path)
        return None if value is MISSING else value

    @property
    def is_present(self) -> bool:
        """Whether the actual document carries this node at all.

        `actual` reports both cases as `None`; these are not the same claim:

        - `{"city": null}` — present. The output says there is no city.
        - `{}` — absent. The output says nothing, and the node exists only
          because `expected` has one.
        """
        return navigate(self.context.actual, self.path) is not MISSING

    @property
    def representative(self) -> float:
        """The node's single representative score: its `key_metric`'s value.

        Parents aggregate their children's representatives post-order, so by the
        time anyone reads this the value exists. The root's is `report.score`.

        Raises:
            ValueError: If the node has no `key_metric`, or its value was never
                computed. Both are programming errors, not a fallback path.
        """
        km = self.key_metric
        if km is None:
            raise ValueError(f"node {self.path!r} has no key_metric")
        value = self.metric_results.get(km.name)
        if value is None:
            raise ValueError(
                f"node {self.path!r}: key_metric {km.name!r} has no computed value"
            )
        return float(value)

    # ── traversal ──────────────────────────────────────────────────────────
    # Children are discovered by duck-typing (`children` on objects, `items`
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
        return (
            getattr(self, "children", None) is None
            and getattr(self, "items", None) is None
        )

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
