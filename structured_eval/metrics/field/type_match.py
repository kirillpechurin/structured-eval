from __future__ import annotations

from typing import Any

from structured_eval.metrics.protocol import FieldMetric


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

    Catches a common LLM error — emitting ``"100"`` (string) where ``100``
    (number) is expected — independently of value correctness.
    """

    name = "type_match"

    def score(self, actual: Any, expected: Any) -> float:
        return 1.0 if _json_type(actual) == _json_type(expected) else 0.0
