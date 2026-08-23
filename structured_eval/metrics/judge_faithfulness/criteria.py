"""Which criterion applies to which field — the path side of the judge's config.

The caller writes criteria against *the node being judged*, and the metric
has to line those up with the leaves it actually found.
"""

from __future__ import annotations

import re

ROOT = "$"  # the document root's path, as the engine writes it


def relative_path(root: str, path: str) -> str:
    """`path` as seen from the node at `root` (`vendor.name` → `name`).

    The judged node itself is named rather than pointed at, since a relative
    path to oneself would be empty and an empty label tells the model nothing
    about what it is grading.

    Args:
        root: The node's absolute path.
        path: The absolute path to rewrite.

    Returns:
        The path as seen from `root`.
    """
    if path == root:
        return path.rsplit(".", 1)[-1]
    if root in (ROOT, ""):
        return path
    if path.startswith(f"{root}."):
        return path[len(root) + 1 :]
    if path.startswith(f"{root}["):
        return path[len(root) :]
    return path


def _pattern(key: str) -> re.Pattern[str] | None:
    r"""Compile a `[*]`-bearing key into a matcher; `None` for a literal key.

    The literal parts are escaped — a path is full of characters a regex would
    otherwise read as syntax (`.`, `[`, `]`) — and each wildcard becomes "one
    array index".

    Args:
        key: A criteria key, possibly bearing `[*]` wildcards.

    Returns:
        A matcher turning `"line_items[*].sku"` into `^line_items\[\d+\]\.sku$`,
        or `None` when the key holds no wildcard.
    """
    if "[*]" not in key:
        return None
    literals = [re.escape(part) for part in key.split("[*]")]
    return re.compile(rf"^{r'\[\d+\]'.join(literals)}$")


class Criteria:
    """What "faithfulness" means for each field, resolved by relative path.

    Accepts the two shapes a caller writes:

    - a bare string — one rule for every field beneath the judged node;
    - a mapping from relative path to rule, where a field with no entry falls
      back to `default`.
    """

    def __init__(self, spec: str | dict[str, str] | None, default: str) -> None:
        """Resolve the caller's spec into a lookup.

        Args:
            spec: A bare criterion for every field, a mapping from relative
                path to criterion, or `None` to use `default` throughout.
            default: The criterion for fields the mapping does not name.
        """
        self.default = spec if isinstance(spec, str) else default
        self.by_path: dict[str, str] = dict(spec) if isinstance(spec, dict) else {}
        self.patterns = [
            (pattern, criterion)
            for key, criterion in self.by_path.items()
            if (pattern := _pattern(key)) is not None
        ]

    @property
    def addresses_fields(self) -> bool:
        """Whether the caller named fields at all, rather than one rule for all.

        A mapping on a leaf is a configuration error — a leaf has no fields for
        the keys to point at — while a bare string is meaningful anywhere.
        """
        return bool(self.by_path)

    def for_path(self, path: str) -> str:
        """The criterion for one field, by its path relative to the judged node.

        A literal key wins over a wildcard, so a specific `"tags[0]"` sits above
        a general `"tags[*]"` whatever order they were written in.

        Args:
            path: The field's relative path.

        Returns:
            The criterion that applies to it.
        """
        if (exact := self.by_path.get(path)) is not None:
            return exact
        for pattern, criterion in self.patterns:
            if pattern.match(path):
                return criterion
        return self.default
