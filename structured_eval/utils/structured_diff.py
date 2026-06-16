from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DiffType(StrEnum):
    ADDED = "added"  # present in actual, absent in expected
    REMOVED = "removed"  # present in expected, absent in actual
    CHANGED = "changed"  # present in both but value differs


class DiffEntry(BaseModel):
    """Single difference between actual and expected at one field path."""

    path: str = Field(description="Dot/bracket path to the differing field.")
    diff_type: DiffType = Field(description="Type of difference: added, removed, or changed.")
    actual: Any = Field(description="Value in actual (None for removed entries).")
    expected: Any = Field(description="Value in expected (None for added entries).")


class StructuredDiff(BaseModel):
    """Human-readable field-level diff between actual and expected documents.

    Produced by structured_diff(). Use .added, .removed, .changed for
    filtered views, or .is_equal to check whether the documents match.
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

    Uses DeepDiff to detect changes at every nesting level and converts
    the result into DiffEntry objects with ADDED / REMOVED / CHANGED types.

    Args:
        actual: LLM output document.
        expected: Ground truth document.

    Returns:
        StructuredDiff with one DiffEntry per differing field path.

    Raises:
        ImportError: If deepdiff is not installed.
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
    """Convert DeepDiff path notation to dot/bracket notation.

    root['a']['b'][0] → a.b[0]
    """
    path = deepdiff_path[4:]  # strip leading "root"
    path = re.sub(r"\['([^']+)'\]", r".\1", path)
    return path.lstrip(".")
