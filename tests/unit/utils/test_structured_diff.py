"""structured_diff — added/removed/changed entries over two documents."""

import pytest

from structured_eval.utils.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    _to_readable_path,
    structured_diff,
)

pytestmark = pytest.mark.unit


# ── _to_readable_path ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "readable"),
    [
        ("root['status']", "status"),
        ("root['b']['c']", "b.c"),
        ("root['lst'][2]", "lst[2]"),
        ("root['items'][1]['x']", "items[1].x"),
    ],
    ids=["top", "nested", "index", "index-nested"],
)
def test_to_readable_path(raw, readable):
    assert _to_readable_path(raw) == readable


# ── equality ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "doc",
    [{"a": 1, "b": "hello"}, {"invoice": {"id": "1", "total": 100.0}}, {}],
    ids=["flat", "nested", "empty"],
)
def test_identical_documents_are_equal(doc):
    diff = structured_diff(doc, doc)
    assert diff.is_equal
    assert diff.entries == []


# ── added ─────────────────────────────────────────────────────────────────────


def test_top_level_added():
    diff = structured_diff({"status": "paid", "extra": "bonus"}, {"status": "paid"})
    (entry,) = diff.added
    assert entry.path == "extra"
    assert entry.diff_type == DiffType.ADDED
    assert entry.actual == "bonus"
    assert entry.expected is None


def test_nested_added():
    diff = structured_diff({"invoice": {"id": "1", "note": "new"}}, {"invoice": {"id": "1"}})
    assert [e.path for e in diff.added] == ["invoice.note"]


def test_list_item_added():
    diff = structured_diff({"items": [1, 2, 3]}, {"items": [1, 2]})
    (entry,) = diff.added
    assert entry.path == "items[2]"
    assert entry.actual == 3


# ── removed ───────────────────────────────────────────────────────────────────


def test_top_level_removed():
    diff = structured_diff({"status": "paid"}, {"status": "paid", "required_field": "value"})
    (entry,) = diff.removed
    assert entry.path == "required_field"
    assert entry.diff_type == DiffType.REMOVED
    assert entry.actual is None
    assert entry.expected == "value"


def test_nested_removed():
    diff = structured_diff({"invoice": {"id": "1"}}, {"invoice": {"id": "1", "tax": 10}})
    assert [e.path for e in diff.removed] == ["invoice.tax"]


def test_list_item_removed():
    diff = structured_diff({"items": [1]}, {"items": [1, 2]})
    (entry,) = diff.removed
    assert entry.path == "items[1]"
    assert entry.expected == 2


# ── changed ───────────────────────────────────────────────────────────────────


def test_top_level_changed():
    diff = structured_diff({"status": "paid"}, {"status": "draft"})
    (entry,) = diff.changed
    assert entry.path == "status"
    assert entry.diff_type == DiffType.CHANGED
    assert entry.actual == "paid"
    assert entry.expected == "draft"


def test_nested_changed():
    diff = structured_diff({"invoice": {"total": 100}}, {"invoice": {"total": 200}})
    (entry,) = diff.changed
    assert entry.path == "invoice.total"
    assert entry.actual == 100
    assert entry.expected == 200


def test_array_item_changed():
    diff = structured_diff(
        {"items": [{"name": "Widget"}, {"name": "Gadget"}]},
        {"items": [{"name": "Widget"}, {"name": "Gadget2"}]},
    )
    assert [e.path for e in diff.changed] == ["items[1].name"]


def test_type_change_counts_as_changed():
    diff = structured_diff({"count": "5"}, {"count": 5})
    (entry,) = diff.changed
    assert entry.diff_type == DiffType.CHANGED
    assert entry.actual == "5"
    assert entry.expected == 5


# ── mixed ─────────────────────────────────────────────────────────────────────


def test_multiple_diff_types_at_once():
    diff = structured_diff({"a": 1, "b": 2, "c": "new"}, {"a": 1, "b": 99, "d": "old"})
    assert [e.path for e in diff.changed] == ["b"]
    assert [e.path for e in diff.added] == ["c"]
    assert [e.path for e in diff.removed] == ["d"]


def test_entries_sorted_by_path():
    diff = structured_diff({"z": 1, "a": 2, "m": 3}, {"z": 2, "a": 3, "m": 4})
    paths = [e.path for e in diff.entries]
    assert paths == sorted(paths)


def test_equal_nested_has_no_entries():
    doc = {"x": 1, "y": {"z": [1, 2, 3]}}
    assert structured_diff(doc, doc).entries == []


# ── StructuredDiff model ──────────────────────────────────────────────────────


def test_empty_diff_is_equal():
    assert StructuredDiff().is_equal is True


def test_diff_with_entries_is_not_equal():
    entry = DiffEntry(path="x", diff_type=DiffType.ADDED, actual=1, expected=None)
    assert StructuredDiff(entries=[entry]).is_equal is False


def test_property_filters_by_type():
    entries = [
        DiffEntry(path="a", diff_type=DiffType.ADDED, actual=1, expected=None),
        DiffEntry(path="b", diff_type=DiffType.REMOVED, actual=None, expected=2),
        DiffEntry(path="c", diff_type=DiffType.CHANGED, actual=3, expected=4),
    ]
    diff = StructuredDiff(entries=entries)
    assert [e.path for e in diff.added] == ["a"]
    assert [e.path for e in diff.removed] == ["b"]
    assert [e.path for e in diff.changed] == ["c"]


def test_diff_type_string_values():
    assert DiffType.ADDED == "added"
    assert DiffType.REMOVED == "removed"
    assert DiffType.CHANGED == "changed"


def test_importable_from_top_level():
    from structured_eval import DiffEntry, DiffType, StructuredDiff, structured_diff  # noqa: F401
