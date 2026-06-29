from structured_eval.utils import flatten
from structured_eval.utils.flatten import flatten as flatten_direct

# ── Basic cases ───────────────────────────────────────────────────────────────


def test_flat_dict() -> None:
    assert flatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_empty_dict() -> None:
    assert flatten({}) == {}


def test_single_key() -> None:
    assert flatten({"x": "hello"}) == {"x": "hello"}


# ── Nested dicts ──────────────────────────────────────────────────────────────


def test_one_level_nested() -> None:
    assert flatten({"a": {"b": 1}}) == {"a.b": 1}


def test_two_levels_nested() -> None:
    assert flatten({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}


def test_mixed_depth() -> None:
    result = flatten({"id": "1", "address": {"city": "Berlin", "zip": "10115"}})
    assert result == {"id": "1", "address.city": "Berlin", "address.zip": "10115"}


def test_multiple_nested_keys() -> None:
    result = flatten({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    assert result == {"a.x": 1, "a.y": 2, "b.z": 3}


# ── Lists ─────────────────────────────────────────────────────────────────────


def test_list_of_primitives() -> None:
    assert flatten({"tags": [1, 2, 3]}) == {"tags[0]": 1, "tags[1]": 2, "tags[2]": 3}


def test_list_of_strings() -> None:
    assert flatten({"names": ["Alice", "Bob"]}) == {"names[0]": "Alice", "names[1]": "Bob"}


def test_list_of_dicts() -> None:
    result = flatten({"items": [{"name": "A", "price": 10}, {"name": "B", "price": 20}]})
    assert result == {
        "items[0].name": "A",
        "items[0].price": 10,
        "items[1].name": "B",
        "items[1].price": 20,
    }


def test_nested_list_in_nested_dict() -> None:
    result = flatten({"invoice": {"items": [{"sku": "X1"}]}})
    assert result == {"invoice.items[0].sku": "X1"}


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_dict_as_value() -> None:
    # Empty dict → kept as leaf value
    assert flatten({"a": {}}) == {"a": {}}


def test_empty_list_as_value() -> None:
    # Empty list → kept as leaf value
    assert flatten({"a": []}) == {"a": []}


def test_none_value() -> None:
    assert flatten({"a": None}) == {"a": None}


def test_bool_value() -> None:
    assert flatten({"flag": True}) == {"flag": True}


def test_zero_value() -> None:
    assert flatten({"count": 0}) == {"count": 0}


def test_false_value() -> None:
    assert flatten({"active": False}) == {"active": False}


# ── Top-level list ────────────────────────────────────────────────────────────


def test_top_level_list_of_dicts() -> None:
    result = flatten([{"name": "A"}, {"name": "B"}])
    assert result == {"[0].name": "A", "[1].name": "B"}


def test_top_level_list_of_primitives() -> None:
    result = flatten([1, 2, 3])
    assert result == {"[0]": 1, "[1]": 2, "[2]": 3}


def test_top_level_empty_list() -> None:
    assert flatten([]) == {}


# ── Import paths ──────────────────────────────────────────────────────────────


def test_import_from_utils_package() -> None:
    # accessible as: from structured_eval.utils import flatten
    assert flatten is flatten_direct


# ── Real-world example ────────────────────────────────────────────────────────


def test_invoice_example() -> None:
    doc = {
        "invoice": {
            "id": "INV-001",
            "vendor": {"name": "Acme Corp", "country": "US"},
            "items": [
                {"sku": "A1", "price": 100.0},
                {"sku": "B2", "price": 200.0},
            ],
            "total": 300.0,
        }
    }
    result = flatten(doc)
    assert result["invoice.id"] == "INV-001"
    assert result["invoice.vendor.name"] == "Acme Corp"
    assert result["invoice.vendor.country"] == "US"
    assert result["invoice.items[0].sku"] == "A1"
    assert result["invoice.items[0].price"] == 100.0
    assert result["invoice.items[1].sku"] == "B2"
    assert result["invoice.total"] == 300.0
