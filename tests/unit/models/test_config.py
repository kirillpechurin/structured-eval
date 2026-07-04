"""Config models — defaults, nesting, and enum value sets (model/config.py)."""

from typing import Any

import pytest

from structured_eval.metrics import ExactMatch, ObjectF1
from structured_eval.models import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    NullPolicy,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.unit


# ── defaults ─────────────────────────────────────────────────────────────────


def test_eval_config_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.metrics == []
    assert cfg.fields == {}
    assert cfg.null_policy == NullPolicy.PENALIZE
    assert cfg.extra_keys == ExtraKeysPolicy.IGNORE
    assert cfg.key_metric is None


def test_field_config_defaults() -> None:
    fc = FieldConfig()
    assert fc.weight == 1.0
    assert fc.required is False
    assert fc.null_policy is None  # inherit


def test_array_config_defaults() -> None:
    ac = ArrayFieldConfig()
    assert ac.strategy == ArrayStrategy.BY_INDEX
    assert ac.params == {}


# ── metric holders ───────────────────────────────────────────────────────────


def test_eval_config_holds_metric_instances() -> None:
    cfg = EvalConfig(metrics=[ObjectF1()], key_metric=ObjectF1())
    assert isinstance(cfg.metrics[0], ObjectF1)


def test_field_config_holds_metric_list() -> None:
    fc = FieldConfig(metrics=[ExactMatch()], key_metric=ExactMatch())
    assert fc.metrics is not None
    assert isinstance(fc.metrics[0], ExactMatch)


# ── nesting ──────────────────────────────────────────────────────────────────


def test_object_field_config_nests() -> None:
    cfg = ObjectFieldConfig(
        fields={"vendor": ObjectFieldConfig(fields={"name": FieldConfig()})}
    )
    assert isinstance(cfg.fields["vendor"], ObjectFieldConfig)


def test_array_item_config() -> None:
    cfg = ArrayFieldConfig(
        item=ObjectFieldConfig(fields={"id": FieldConfig()}),
        strategy=ArrayStrategy.BY_KEY,
        params={"key": "id"},
    )
    assert cfg.params["key"] == "id"
    assert isinstance(cfg.item, ObjectFieldConfig)


# ── enum value sets ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (NullPolicy, {"ignore", "penalize", "require_match"}),
        (ExtraKeysPolicy, {"ignore", "penalize"}),
        (ArrayStrategy, {"by_index", "by_key", "hungarian"}),
    ],
    ids=["null-policy", "extra-keys", "array-strategy"],
)
def test_enum_values(enum: Any, values: Any) -> None:
    assert {member.value for member in enum} == values
