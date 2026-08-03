"""Which criterion applies to which field — the path side of the judge's config.

Kept apart from ``prompt`` (what the judge is asked) and ``metric`` (the
bookkeeping) because this is its own question: the caller writes criteria
against *the node being judged*, and the metric has to line those up with the
leaves it actually found.

Paths here are **relative to the judged node**, so one configuration works
wherever the judge is hung — ``{"name": ...}`` grades ``vendor.name`` when the
judge sits on ``vendor`` and ``name`` when it sits on the document root. A
judge hung on a single field addresses it by its own name (``vendor.city`` →
``city``): the judge has to see *what* it is grading, and a field's name is
usually the only clue the model gets besides the value.

``[*]`` stands for any array index, the wildcard the rest of the config uses:
``"line_items[*].sku"`` grades the ``sku`` of every element. A literal index
(``"tags[0]"``) also works and wins over a wildcard — for the rare case where
one position genuinely differs from its neighbours.
"""

from __future__ import annotations

import re

ROOT = "$"  # the document root's path, as the engine writes it


def relative_path(root: str, path: str) -> str:
    """``path`` as seen from the node at ``root`` (``vendor.name`` → ``name``).

    ``root`` is the judged node's absolute path. The document root is ``"$"``
    and its children carry no prefix, so relative and absolute coincide there.
    An array index follows its parent without a dot, hence the two prefixes.

    The judged node itself is named rather than pointed at: ``vendor.city``
    judged on its own is ``city``. A relative path to oneself would be empty,
    and an empty label tells the model nothing about what it is grading.
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
    """Compile a ``[*]``-bearing key into a matcher; ``None`` for a literal key.

    ``"line_items[*].sku"`` becomes ``^line_items\\[\\d+\\]\\.sku$``: the literal
    parts are escaped — a path is full of characters a regex would otherwise
    read as syntax (``.``, ``[``, ``]``) — and each wildcard becomes "one array
    index".

    Splitting on the wildcard *before* escaping is what keeps this readable.
    Escaping the whole key first and substituting afterwards would mean
    searching an already-escaped string for an already-escaped needle.
    """
    if "[*]" not in key:
        return None
    literals = [re.escape(part) for part in key.split("[*]")]
    return re.compile(rf"^{r'\[\d+\]'.join(literals)}$")


class Criteria:
    """What "faithful" means for each field, resolved by relative path.

    Accepts the two shapes a caller writes: a bare string — one rule for every
    field beneath the judged node — or a mapping from relative path to rule,
    where fields with no entry fall back to ``default``.

    ``addresses_fields`` reports whether the caller named fields at all: a
    mapping on a leaf is a configuration error (a leaf has no fields for the
    keys to point at), while a bare string is meaningful anywhere.
    """

    def __init__(self, spec: str | dict[str, str] | None, default: str) -> None:
        self.default = spec if isinstance(spec, str) else default
        self.by_path: dict[str, str] = dict(spec) if isinstance(spec, dict) else {}
        self.patterns = [
            (pattern, criterion)
            for key, criterion in self.by_path.items()
            if (pattern := _pattern(key)) is not None
        ]

    @property
    def addresses_fields(self) -> bool:
        """Whether the criteria are keyed by path rather than one rule for all."""
        return bool(self.by_path)

    def for_path(self, path: str) -> str:
        """The criterion for one field, by its path relative to the judged node.

        A literal key wins over a wildcard, so a specific ``"tags[0]"`` sits
        above a general ``"tags[*]"`` whatever order they were written in.
        """
        if (exact := self.by_path.get(path)) is not None:
            return exact
        for pattern, criterion in self.patterns:
            if pattern.match(path):
                return criterion
        return self.default
