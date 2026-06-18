"""Path navigation over nested dict/list documents.

A pure utility (no model dependencies) shared by the node tree, the engine's
tree builder, and array alignment. ``navigate`` walks a dot-and-bracket path;
``MISSING`` is the sentinel for an unresolvable step (distinct from ``None`` so
callers can tell "absent" from "present but null").
"""

from __future__ import annotations

import re
from typing import Any

# Splits a dot-and-bracket path into access steps.
#   "roles[0].name" → ["roles", "[0]", "name"]
#   "items[0]"      → ["items", "[0]"]
#   "a.b"           → ["a", "b"]
_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[[^\]]*\]")

# Sentinel returned when a path cannot be resolved (key/index missing).
# Distinct from None so callers can tell "absent" from "present but null".
MISSING = object()


def navigate(obj: Any, path: str) -> Any:
    """Walk ``obj`` along a dot-and-bracket ``path``.

    ``"$"`` returns the root unchanged. Dict keys use dot notation, list
    indices use brackets: ``"roles[0].name"``. Returns ``MISSING`` when any
    step cannot be resolved (missing key, out-of-range or non-integer index).

    Examples:
        >>> navigate({"a": {"b": 1}}, "a.b")
        1
        >>> navigate({"items": [1, 2]}, "items[0]")
        1
    """
    if path == "$" or path == "":
        return obj

    current = obj
    for token in _PATH_TOKEN.findall(path):
        if token.startswith("["):
            inner = token[1:-1]
            if not isinstance(current, list):
                return MISSING
            try:
                idx = int(inner)
            except ValueError:
                return MISSING
            if not -len(current) <= idx < len(current):
                return MISSING
            current = current[idx]
        else:
            if not isinstance(current, dict) or token not in current:
                return MISSING
            current = current[token]
    return current
