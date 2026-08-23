"""The `type_match` metric — do actual and expected share a JSON type?"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric


def _json_type(value: Any) -> str:
    """Map a Python value to its JSON type name (bool before int)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


class TypeMatch(FieldMetric):
    """Right JSON type? 1.0 if actual and expected share a type, else 0.0.

    Catches a common LLM error — emitting `"100"` where `100` is expected —
    independently of whether the value itself is right.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import TypeMatch
        >>> from structured_eval.models import EvalConfig
        >>> TypeMatch().score(100, 42)          # both numbers
        1.0
        >>> TypeMatch().score("100", 100)       # string vs number
        0.0
        >>> report = evaluate({"total": "100", "qty": 3},
        ...                   {"total": 100, "qty": 7},
        ...                   EvalConfig(metrics=[TypeMatch()]))
        >>> float(report.metrics["type_match"].representative())
        0.5
    """

    name = "type_match"

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when both values map to the same JSON type, else 0.0."""
        return 1.0 if _json_type(actual) == _json_type(expected) else 0.0
