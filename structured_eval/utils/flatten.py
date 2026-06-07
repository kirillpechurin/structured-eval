from __future__ import annotations

from typing import Any


def flatten(obj: dict[str, Any] | list[Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict/list into dot-and-bracket key paths.

    Dict keys use dot notation:     {"a": {"b": 1}}         → {"a.b": 1}
    List indices use brackets:      {"a": [1, 2]}           → {"a[0]": 1, "a[1]": 2}
    Empty containers are left as-is: {"a": {}}              → {"a": {}}

    Args:
        obj: Dict or list to flatten.
        prefix: Internal prefix for recursive calls; do not pass externally.

    Returns:
        Flat dict mapping string paths to primitive (or empty container) values.

    Example:
        >>> flatten({"invoice": {"id": "1", "items": [{"price": 100}]}})
        {"invoice.id": "1", "invoice.items[0].price": 100}
    """
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)) and value:
                result.update(flatten(value, path))
            else:
                result[path] = value
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)) and item:
                result.update(flatten(item, path))
            else:
                result[path] = item
    return result
