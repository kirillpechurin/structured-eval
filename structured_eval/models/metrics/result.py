"""`MetricResult` — a metric's value together with its structured detail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler


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
