"""`MetricResult` — a metric's value together with its structured detail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler


class MetricResult(float):
    """A metric value: a `float` everywhere, plus structured `.extra`.

    Attributes:
        extra: The metric's structured detail; empty when it published none.

    Example:
        >>> from structured_eval.models import MetricResult
        >>> score = MetricResult(0.75, extra={"tp": 3, "fp": 1, "fn": 0})
        >>> score < 1.0  # a plain float wherever a number is wanted
        True
        >>> round(score * 4)
        3
        >>> score.extra["fp"]
        1
        >>> MetricResult(1.0)  # no detail published, no `extra` shown
        MetricResult(1.0)
    """

    extra: dict[str, Any]

    def __new__(cls, value: float, extra: dict[str, Any] | None = None) -> MetricResult:
        """Build the value, copying `extra` so the caller's dict stays its own."""
        obj = super().__new__(cls, value)
        obj.extra = dict(extra) if extra else {}
        return obj

    def __repr__(self) -> str:
        """Show the number, and `extra` only when there is any."""
        num = float.__repr__(self)
        return (
            f"MetricResult({num}, extra={self.extra!r})"
            if self.extra
            else f"MetricResult({num})"
        )

    # ── pydantic (round-trips extra: serialized as a bare float when empty,
    #    else as `{"value": ..., "extra": ...}`; both forms re-validate) ──
    @classmethod
    def _validate(cls, value: Any) -> MetricResult:
        """Either serialized form read back; an instance passes through."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value["value"], value.get("extra"))
        return cls(value)

    @staticmethod
    def _serialize(value: MetricResult) -> Any:
        """A bare float when there is no `extra`, the pair otherwise."""
        return (
            {"value": float(value), "extra": value.extra}
            if value.extra
            else float(value)
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """The pydantic schema that round-trips the value and its `extra`."""
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            ),
        )
