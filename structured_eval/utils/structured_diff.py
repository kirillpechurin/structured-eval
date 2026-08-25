"""A readable field-level diff between an actual and an expected document."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DiffType(StrEnum):
    """Which side of the comparison a difference falls on.

    Attributes:
        ADDED: Present in actual, absent in expected.
        REMOVED: Present in expected, absent in actual.
        CHANGED: Present in both, but the values differ.
    """

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class DiffEntry(BaseModel):
    """Single difference between actual and expected at one field path.

    Attributes:
        path: Dot-and-bracket path to the differing field.
        diff_type: Which side of the comparison the difference falls on.
        actual: Value in actual; `None` for a removed entry.
        expected: Value in expected; `None` for an added entry.
    """

    path: str
    diff_type: DiffType
    actual: Any
    expected: Any


class StructuredDiff(BaseModel):
    """Human-readable field-level diff between actual and expected documents.

    What `structured_diff` returns. `added` / `removed` / `changed` are filtered
    views of the same entries, and `is_equal` says whether there are any.

    Attributes:
        entries: One entry per differing field path, ordered by path.
    """

    entries: list[DiffEntry] = Field(default_factory=list)

    @property
    def added(self) -> list[DiffEntry]:
        """Fields present in actual but absent in expected."""
        return [e for e in self.entries if e.diff_type == DiffType.ADDED]

    @property
    def removed(self) -> list[DiffEntry]:
        """Fields present in expected but absent in actual."""
        return [e for e in self.entries if e.diff_type == DiffType.REMOVED]

    @property
    def changed(self) -> list[DiffEntry]:
        """Fields present in both but with different values."""
        return [e for e in self.entries if e.diff_type == DiffType.CHANGED]

    @property
    def is_equal(self) -> bool:
        """True when actual and expected are identical (no differences)."""
        return len(self.entries) == 0


def structured_diff(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> StructuredDiff:
    """Compute a readable field-level diff between actual and expected.

    Uses DeepDiff to detect changes at every nesting level and converts the
    result into `DiffEntry` objects with dot-and-bracket paths. Requires the
    `diff` extra.

    Args:
        actual: LLM output document.
        expected: Ground truth document.

    Returns:
        A `StructuredDiff` with one entry per differing field path, sorted by
        path.

    Raises:
        ImportError: If deepdiff is not installed.

    Example:
        >>> from structured_eval.utils import structured_diff
        >>> diff = structured_diff({"total": 120, "tax": 20},
        ...                        {"total": 100, "currency": "EUR"})
        >>> [(e.path, e.diff_type.value) for e in diff.entries]
        [('currency', 'removed'), ('tax', 'added'), ('total', 'changed')]
        >>> diff.changed[0].actual, diff.changed[0].expected
        (120, 100)
        >>> diff.is_equal
        False
    """
    try:
        from deepdiff import DeepDiff
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "deepdiff is required for structured_diff. "
            "Install it with: pip install 'structured-eval[diff]'"
        ) from exc

    # DeepDiff(old, new) — expected is old, actual is new
    diff = DeepDiff(expected, actual, verbose_level=2)
    entries: list[DiffEntry] = []

    for raw_path, value in diff.get("dictionary_item_added", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.ADDED,
                actual=value,
                expected=None,
            )
        )

    for raw_path, value in diff.get("dictionary_item_removed", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.REMOVED,
                actual=None,
                expected=value,
            )
        )

    for raw_path, change in diff.get("values_changed", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.CHANGED,
                actual=change["new_value"],
                expected=change["old_value"],
            )
        )

    for raw_path, change in diff.get("type_changes", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.CHANGED,
                actual=change["new_value"],
                expected=change["old_value"],
            )
        )

    for raw_path, value in diff.get("iterable_item_added", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.ADDED,
                actual=value,
                expected=None,
            )
        )

    for raw_path, value in diff.get("iterable_item_removed", {}).items():
        entries.append(
            DiffEntry(
                path=_to_readable_path(raw_path),
                diff_type=DiffType.REMOVED,
                actual=None,
                expected=value,
            )
        )

    entries.sort(key=lambda e: e.path)
    return StructuredDiff(entries=entries)


def _to_readable_path(deepdiff_path: str) -> str:
    """Convert a DeepDiff path to dot/bracket form: `root['a'][0]` → `a[0]`."""
    path = deepdiff_path[4:]  # strip leading "root"
    path = re.sub(r"\['([^']+)'\]", r".\1", path)
    return path.lstrip(".")
