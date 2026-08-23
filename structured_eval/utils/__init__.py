"""Document helpers shared by the layers above.

- `flatten` / `extract_paths` — a nested document as dot-and-bracket field paths.
- `structured_diff` — a readable field-level diff between two documents.
"""

from structured_eval.utils.flatten import extract_paths, flatten
from structured_eval.utils.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    structured_diff,
)

__all__ = [
    "DiffEntry",
    "DiffType",
    "StructuredDiff",
    "extract_paths",
    "flatten",
    "structured_diff",
]
