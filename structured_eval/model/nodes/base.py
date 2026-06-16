from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.model.context import EvalContext

# Splits a dot-and-bracket path into access steps.
#   "roles[0].name" → ["roles", "[0]", "name"]
#   "items[0]"      → ["items", "[0]"]
#   "a.b"           → ["a", "b"]
_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[[^\]]*\]")

# Sentinel returned when a path cannot be resolved (key/index missing).
# Distinct from None so callers can tell "absent" from "present but null".
MISSING = object()


def _navigate(obj: Any, path: str) -> Any:
    """Walk ``obj`` along a dot-and-bracket ``path``.

    ``"$"`` returns the root unchanged. Dict keys use dot notation, list
    indices use brackets: ``"roles[0].name"``. Returns ``MISSING`` when any
    step cannot be resolved (missing key, out-of-range or non-integer index).

    Examples:
        >>> _navigate({"a": {"b": 1}}, "a.b")
        1
        >>> _navigate({"items": [1, 2]}, "items[0]")
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


class EvalNode(BaseModel):
    """A node in the evaluation tree.

    Holds its ``path`` and a shared reference to the ``EvalContext``; data is
    never copied — ``actual``/``expected`` are resolved lazily by navigating the
    context's documents. ``expected_path`` defaults to ``path``; it diverges
    only for array items aligned out of order (``expected[1]`` ↔ ``actual[0]``),
    so each side navigates its own index. ``metric_results`` accumulates each
    requested metric's value at this node (filled by the engine in phase 2).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    context: EvalContext
    expected_path: str | None = None
    weight: float = 1.0  # relative importance for weighted aggregation (OverallScore)
    metric_results: dict[str, float] = Field(default_factory=dict)

    @property
    def actual(self) -> Any:
        value = _navigate(self.context.actual, self.path)
        return None if value is MISSING else value

    @property
    def expected(self) -> Any:
        if self.context.expected is None:
            return None
        value = _navigate(self.context.expected, self.expected_path or self.path)
        return None if value is MISSING else value

    # ── traversal ──────────────────────────────────────────────────────────
    # Children are discovered by duck-typing (``children`` on objects, ``items``
    # on arrays) so the base node need not import its own subclasses.

    def children_nodes(self) -> Iterator[EvalNode]:
        """Yield the node's direct child nodes (none for a scalar leaf)."""
        children = getattr(self, "children", None)
        if isinstance(children, dict):
            yield from children.values()
        items = getattr(self, "items", None)
        if isinstance(items, list):
            yield from items

    def is_leaf(self) -> bool:
        """True for a scalar node (no object children, no array items)."""
        return getattr(self, "children", None) is None and getattr(self, "items", None) is None

    def walk(self) -> Iterator[EvalNode]:
        """Depth-first traversal yielding this node and every descendant."""
        yield self
        for child in self.children_nodes():
            yield from child.walk()

    def leaves(self) -> Iterator[EvalNode]:
        """Yield every scalar (leaf) node at or beneath this node."""
        for node in self.walk():
            if node.is_leaf():
                yield node
