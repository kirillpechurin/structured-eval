"""Config models — defaults, nesting, and enum value sets (model/config.py)."""

from typing import Any

import pytest

from structured_eval.metrics import ArrayAccuracy, ExactMatch, ObjectF1
from structured_eval.models import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.unit


# ── defaults ─────────────────────────────────────────────────────────────────


def test_eval_config_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.metrics == []
    assert cfg.fields == {}
    assert cfg.extra_keys == ExtraKeysPolicy.IGNORE
    assert cfg.key_metric is None


def test_field_config_defaults() -> None:
    fc = FieldConfig()
    assert fc.weight == 1.0


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


def test_object_and_array_config_hold_key_metric() -> None:
    # #49: the representative-metric override, symmetric with FieldConfig.
    oc = ObjectFieldConfig(key_metric=ObjectF1())
    ac = ArrayFieldConfig(key_metric=ArrayAccuracy())
    assert isinstance(oc.key_metric, ObjectF1)
    assert isinstance(ac.key_metric, ArrayAccuracy)


def test_array_representative_key_metric_is_distinct_from_by_key_param() -> None:
    # the node's representative key_metric and the BY_KEY alignment key_metric
    # in params are different concepts and coexist independently.
    ac = ArrayFieldConfig(
        key_metric=ArrayAccuracy(),
        strategy=ArrayStrategy.BY_KEY,
        params={"key": "id", "key_metric": ExactMatch()},
    )
    assert isinstance(ac.key_metric, ArrayAccuracy)  # representative
    assert isinstance(ac.params["key_metric"], ExactMatch)  # alignment matcher


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
        (ExtraKeysPolicy, {"ignore", "penalize"}),
        (ArrayStrategy, {"by_index", "by_key", "hungarian"}),
    ],
    ids=["extra-keys", "array-strategy"],
)
def test_enum_values(enum: Any, values: Any) -> None:
    assert {member.value for member in enum} == values
