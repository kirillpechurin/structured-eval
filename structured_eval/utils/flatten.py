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


def extract_paths(value: Any, prefix: str = "") -> set[str]:
    """Every structural path in a JSON-like value — order- and value-blind.

    Yields the path of each container *and* each leaf, so the set captures the
    whole skeleton: dict keys (``a``, ``a.b``), list indices (``a[0]``) and the
    leaf paths beneath them. Values themselves are ignored — only the shape.
    Unlike :func:`flatten`, intermediate container paths are included, not just
    leaves, and the result is a set of paths rather than a path→value mapping.

    Example:
        >>> sorted(extract_paths({"a": {"b": 1}, "c": [2]}))
        ['a', 'a.b', 'c', 'c[0]']
    """
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            paths.add(here)
            paths |= extract_paths(child, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            here = f"{prefix}[{index}]"
            paths.add(here)
            paths |= extract_paths(child, here)
    return paths
