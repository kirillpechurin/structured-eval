from structured_eval.utils.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    _to_readable_path,
    structured_diff,
)

# ── _to_readable_path ─────────────────────────────────────────────────────────


class TestToReadablePath:
    def test_top_level_key(self):
        assert _to_readable_path("root['status']") == "status"

    def test_nested_key(self):
        assert _to_readable_path("root['b']['c']") == "b.c"

    def test_array_index(self):
        assert _to_readable_path("root['lst'][2]") == "lst[2]"

    def test_array_nested_key(self):
        assert _to_readable_path("root['items'][1]['x']") == "items[1].x"


# ── structured_diff ───────────────────────────────────────────────────────────


class TestStructuredDiffEqual:
    def test_identical_flat(self):
        doc = {"a": 1, "b": "hello"}
        diff = structured_diff(doc, doc)
        assert diff.is_equal
        assert diff.entries == []

    def test_identical_nested(self):
        doc = {"invoice": {"id": "1", "total": 100.0}}
        diff = structured_diff(doc, doc)
        assert diff.is_equal

    def test_empty_dicts(self):
        diff = structured_diff({}, {})
        assert diff.is_equal


class TestStructuredDiffAdded:
    def test_top_level_added(self):
        diff = structured_diff(
            actual={"status": "paid", "extra": "bonus"},
            expected={"status": "paid"},
        )
        assert len(diff.added) == 1
        entry = diff.added[0]
        assert entry.path == "extra"
        assert entry.diff_type == DiffType.ADDED
        assert entry.actual == "bonus"
        assert entry.expected is None

    def test_nested_added(self):
        diff = structured_diff(
            actual={"invoice": {"id": "1", "note": "new"}},
            expected={"invoice": {"id": "1"}},
        )
        assert len(diff.added) == 1
        assert diff.added[0].path == "invoice.note"

    def test_list_item_added(self):
        diff = structured_diff(
            actual={"items": [1, 2, 3]},
            expected={"items": [1, 2]},
        )
        assert len(diff.added) == 1
        assert diff.added[0].path == "items[2]"
        assert diff.added[0].actual == 3


class TestStructuredDiffRemoved:
    def test_top_level_removed(self):
        diff = structured_diff(
            actual={"status": "paid"},
            expected={"status": "paid", "required_field": "value"},
        )
        assert len(diff.removed) == 1
        entry = diff.removed[0]
        assert entry.path == "required_field"
        assert entry.diff_type == DiffType.REMOVED
        assert entry.actual is None
        assert entry.expected == "value"

    def test_nested_removed(self):
        diff = structured_diff(
            actual={"invoice": {"id": "1"}},
            expected={"invoice": {"id": "1", "tax": 10}},
        )
        assert len(diff.removed) == 1
        assert diff.removed[0].path == "invoice.tax"

    def test_list_item_removed(self):
        diff = structured_diff(
            actual={"items": [1]},
            expected={"items": [1, 2]},
        )
        assert len(diff.removed) == 1
        assert diff.removed[0].path == "items[1]"
        assert diff.removed[0].expected == 2


class TestStructuredDiffChanged:
    def test_top_level_changed(self):
        diff = structured_diff(
            actual={"status": "paid"},
            expected={"status": "draft"},
        )
        assert len(diff.changed) == 1
        entry = diff.changed[0]
        assert entry.path == "status"
        assert entry.diff_type == DiffType.CHANGED
        assert entry.actual == "paid"
        assert entry.expected == "draft"

    def test_nested_changed(self):
        diff = structured_diff(
            actual={"invoice": {"total": 100}},
            expected={"invoice": {"total": 200}},
        )
        assert len(diff.changed) == 1
        assert diff.changed[0].path == "invoice.total"
        assert diff.changed[0].actual == 100
        assert diff.changed[0].expected == 200

    def test_array_item_changed(self):
        diff = structured_diff(
            actual={"items": [{"name": "Widget"}, {"name": "Gadget"}]},
            expected={"items": [{"name": "Widget"}, {"name": "Gadget2"}]},
        )
        assert len(diff.changed) == 1
        assert diff.changed[0].path == "items[1].name"

    def test_type_change_is_changed(self):
        diff = structured_diff(
            actual={"count": "5"},
            expected={"count": 5},
        )
        assert len(diff.changed) == 1
        assert diff.changed[0].diff_type == DiffType.CHANGED
        assert diff.changed[0].actual == "5"
        assert diff.changed[0].expected == 5


class TestStructuredDiffMixed:
    def test_multiple_diff_types(self):
        actual = {"a": 1, "b": 2, "c": "new"}
        expected = {"a": 1, "b": 99, "d": "old"}
        diff = structured_diff(actual, expected)
        assert len(diff.changed) == 1
        assert diff.changed[0].path == "b"
        assert len(diff.added) == 1
        assert diff.added[0].path == "c"
        assert len(diff.removed) == 1
        assert diff.removed[0].path == "d"

    def test_entries_sorted_by_path(self):
        actual = {"z": 1, "a": 2, "m": 3}
        expected = {"z": 2, "a": 3, "m": 4}
        diff = structured_diff(actual, expected)
        paths = [e.path for e in diff.entries]
        assert paths == sorted(paths)

    def test_no_spurious_changed_when_equal(self):
        doc = {"x": 1, "y": {"z": [1, 2, 3]}}
        diff = structured_diff(doc, doc)
        assert len(diff.entries) == 0


# ── StructuredDiff model ──────────────────────────────────────────────────────


class TestStructuredDiffModel:
    def test_is_equal_empty(self):
        assert StructuredDiff().is_equal is True

    def test_is_equal_with_entries(self):
        entry = DiffEntry(path="x", diff_type=DiffType.ADDED, actual=1, expected=None)
        assert StructuredDiff(entries=[entry]).is_equal is False

    def test_property_filters(self):
        entries = [
            DiffEntry(path="a", diff_type=DiffType.ADDED, actual=1, expected=None),
            DiffEntry(path="b", diff_type=DiffType.REMOVED, actual=None, expected=2),
            DiffEntry(path="c", diff_type=DiffType.CHANGED, actual=3, expected=4),
        ]
        diff = StructuredDiff(entries=entries)
        assert [e.path for e in diff.added] == ["a"]
        assert [e.path for e in diff.removed] == ["b"]
        assert [e.path for e in diff.changed] == ["c"]

    def test_diff_type_string_values(self):
        assert DiffType.ADDED == "added"
        assert DiffType.REMOVED == "removed"
        assert DiffType.CHANGED == "changed"


# ── Top-level import ──────────────────────────────────────────────────────────


def test_importable_from_top_level():
    from structured_eval import DiffEntry, DiffType, StructuredDiff, structured_diff  # noqa: F401
