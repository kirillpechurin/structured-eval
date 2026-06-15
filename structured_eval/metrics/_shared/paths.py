from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def extract_paths(obj: Any, prefix: str = "") -> set[str]:
    """Recursively extract all field paths from a nested dict/list.

    Array indices are omitted so that {"items": [{"a": 1}, {"a": 2}]}
    produces {"items", "items.a"} — useful for structural comparison
    regardless of array length.
    """
    paths: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths |= extract_paths(value, path)
    elif isinstance(obj, list):
        for item in obj:
            paths |= extract_paths(item, prefix)
    return paths


def path_recall(actual: Any, expected: Any) -> float:
    """Fraction of expected paths present in actual.

    A low value means the LLM omitted fields that were expected.
    Returns 1.0 when both inputs are empty.
    """
    expected_paths = extract_paths(expected)
    if not expected_paths:
        return 1.0
    actual_paths = extract_paths(actual)
    return len(expected_paths & actual_paths) / len(expected_paths)


def path_precision(actual: Any, expected: Any) -> float:
    """Fraction of actual paths that are present in expected.

    A low value means the LLM added fields that were not expected.
    Returns 1.0 when actual is empty.
    """
    actual_paths = extract_paths(actual)
    if not actual_paths:
        return 1.0
    expected_paths = extract_paths(expected)
    return len(actual_paths & expected_paths) / len(actual_paths)


def coverage_score(actual: Any, schema: Any) -> float:
    """Fraction of schema fields that are present and non-null in actual.

    Supports Pydantic BaseModel subclasses and JSON Schema dicts.
    Only inspects top-level fields; nested objects are not traversed.
    Returns 0.0 if the schema defines no fields.
    """
    schema_fields = _schema_field_names(schema)
    if not schema_fields:
        return 0.0
    if not isinstance(actual, dict):
        return 0.0
    covered = sum(1 for f in schema_fields if actual.get(f) is not None)
    return covered / len(schema_fields)


def _schema_field_names(schema: Any) -> list[str]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return list(schema.model_fields.keys())
    if isinstance(schema, dict):
        return list(schema.get("properties", {}).keys())
    return []
