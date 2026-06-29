"""A metric's value: a float that also carries structured detail.

``MetricResult`` is the single shape every metric value takes once it has passed
through ``MetricRunner._apply`` — whatever a metric's ``compute`` returns (a bare
``float``, a ``dict`` of sub-scores, a ``(value, extra)`` tuple, or a
``MetricResult``) is normalized to it. It *is* a ``float`` (so every existing
numeric use keeps working) and additionally exposes ``.extra`` — arbitrary
structured detail a metric wants to surface beyond the number (offending paths,
per-rule outcomes, an LLM judge's reasoning, …).

``MetricCollection`` is the cross-field view: ``report.metrics[name]`` gathers a
named metric's value at every node that produced it, keyed by path, with numeric
reductions and the union of their ``extra`` payloads.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema


class MetricResult(float):
    """A metric value: a ``float`` everywhere, plus structured ``.extra``."""

    extra: dict[str, Any]

    def __new__(cls, value: float, extra: dict[str, Any] | None = None) -> MetricResult:
        obj = super().__new__(cls, value)
        obj.extra = dict(extra) if extra else {}
        return obj

    def __repr__(self) -> str:
        num = float.__repr__(self)
        return (
            f"MetricResult({num}, extra={self.extra!r})"
            if self.extra
            else f"MetricResult({num})"
        )

    # ── pydantic (round-trips extra: serialized as a bare float when empty,
    #    else as ``{"value": ..., "extra": ...}``; both forms re-validate) ──
    @classmethod
    def _validate(cls, value: Any) -> MetricResult:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value["value"], value.get("extra"))
        return cls(value)

    @staticmethod
    def _serialize(value: MetricResult) -> Any:
        return (
            {"value": float(value), "extra": value.extra}
            if value.extra
            else float(value)
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            ),
        )


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
        """Flatten a list-valued ``extra[key]`` across every node's detail."""
        out: list[Any] = []
        for result in self.values():
            out.extend(result.extra.get(key, []))
        return out
