from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Runtime import: `by_path` is a model field, so pydantic must resolve the type
# to build the schema — under TYPE_CHECKING the model is never fully defined.
from structured_eval.models.metrics.result import MetricResult  # noqa: TC001


class MetricCollection(BaseModel):
    """A named metric's values across the tree (``report.metrics[name]``).

    ``by_path`` maps every node path that produced this metric to its
    ``MetricResult``. Numeric reductions (``mean``/``min``/``max``) summarise the
    whole tree; ``root()`` is the document-level value (path ``"$"``) when the
    metric ran at the root; ``extra`` is the list of non-empty detail payloads.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    by_path: dict[str, MetricResult] = Field(default_factory=dict)

    def values(self) -> list[MetricResult]:
        return list(self.by_path.values())

    def mean(self) -> float:
        vals = self.values()
        return mean(vals) if vals else 0.0

    def min(self) -> float:
        vals = self.values()
        return min(vals) if vals else 0.0

    def max(self) -> float:
        vals = self.values()
        return max(vals) if vals else 0.0

    def root(self) -> MetricResult | None:
        """The document-level value (path ``"$"``), or ``None`` if not at root."""
        return self.by_path.get("$")

    def representative(self) -> float:
        """The document-level value if present, else the mean across the tree."""
        root = self.root()
        return float(root) if root is not None else self.mean()

    @property
    def extra(self) -> list[dict[str, Any]]:
        """The non-empty ``extra`` payloads from each node, in path order."""
        return [r.extra for r in self.values() if r.extra]

    def extra_values(self, key: str) -> list[Any]:
        """Gather ``extra[key]`` across every node's detail.

        A list is flattened into the result, anything else appended whole — so
        a metric publishing one object per node (an LLM judge's verdict) and
        one publishing many items per node (rule results) both read back as a
        flat list.
        """
        out: list[Any] = []
        for result in self.values():
            if isinstance(result.extra.get(key), dict):
                out.append(result.extra[key])
            else:
                out.extend(result.extra.get(key, []))
        return out
