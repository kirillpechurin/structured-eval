"""Unit tests for the config models and their defaults/validation."""

from __future__ import annotations

import pytest

from structured_eval import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExactMatch,
    ExtraKeysPolicy,
    FieldConfig,
    NullPolicy,
    ObjectF1,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.unit


class TestDefaults:
    def test_eval_config_defaults(self):
        cfg = EvalConfig()
        assert cfg.metrics == []
        assert cfg.fields == {}
        assert cfg.null_policy == NullPolicy.PENALIZE
        assert cfg.extra_keys == ExtraKeysPolicy.IGNORE
        assert cfg.key_metric is None

    def test_field_config_defaults(self):
        fc = FieldConfig()
        assert fc.weight == 1.0
        assert fc.required is False
        assert fc.exclude is False
        assert fc.derived is False
        assert fc.null_policy is None  # inherit

    def test_array_config_defaults(self):
        ac = ArrayFieldConfig()
        assert ac.strategy == ArrayStrategy.BY_INDEX
        assert ac.params == {}


class TestArbitraryMetrics:
    def test_holds_metric_instances(self):
        cfg = EvalConfig(metrics=[ObjectF1()], key_metric=ObjectF1())
        assert isinstance(cfg.metrics[0], ObjectF1)

    def test_field_metric_list(self):
        fc = FieldConfig(metrics=[ExactMatch()], key_metric=ExactMatch())
        assert isinstance(fc.metrics[0], ExactMatch)


class TestNesting:
    def test_object_field_config_nests(self):
        cfg = ObjectFieldConfig(
            fields={"vendor": ObjectFieldConfig(fields={"name": FieldConfig()})}
        )
        assert isinstance(cfg.fields["vendor"], ObjectFieldConfig)

    def test_array_item_config(self):
        cfg = ArrayFieldConfig(
            item=ObjectFieldConfig(fields={"id": FieldConfig()}),
            strategy=ArrayStrategy.BY_KEY,
            params={"key": "id"},
        )
        assert cfg.params["key"] == "id"
        assert isinstance(cfg.item, ObjectFieldConfig)


class TestEnums:
    def test_null_policy_values(self):
        assert {p.value for p in NullPolicy} == {"ignore", "penalize", "require_match"}

    def test_extra_keys_values(self):
        assert {p.value for p in ExtraKeysPolicy} == {"ignore", "penalize"}

    def test_array_strategy_values(self):
        assert {s.value for s in ArrayStrategy} == {"by_index", "by_key"}
