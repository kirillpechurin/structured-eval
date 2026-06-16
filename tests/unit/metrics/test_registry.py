"""Contract test for the metric registry.

Every metric exported from the package's public API must register its ``name``
in ``_METRIC_REGISTRY`` and resolve via ``get_metric_class`` — this guards
against the catalog and the registry drifting apart (e.g. EvalConfig.from_yaml).
"""

from __future__ import annotations

import pytest

import structured_eval as se
from structured_eval.metrics.base import (
    _METRIC_REGISTRY,
    Metric,
    get_metric_class,
)

pytestmark = pytest.mark.unit

# Concrete, named metrics in the public catalog (abstract bases excluded).
NAMED_METRICS = [
    "ExactMatch",
    "NormalizedMatch",
    "Numeric",
    "TokenF1",
    "Fuzzy",
    "Presence",
    "TypeMatch",
    "ObjectAccuracy",
    "ObjectPrecision",
    "ObjectRecall",
    "ObjectF1",
    "ObjectPRF1",
    "ObjectValidity",
    "ArrayAccuracy",
    "ArrayPrecision",
    "ArrayRecall",
    "ArrayF1",
    "ArrayPRF1",
    "ArrayCardinality",
    "OverallScore",
    "SchemaValidity",
    "Coverage",
    "Faithfulness",
    "RulePassRate",
]


@pytest.mark.parametrize("cls_name", NAMED_METRICS)
def test_metric_registered(cls_name):
    cls = getattr(se, cls_name)
    assert issubclass(cls, Metric)
    assert cls.name, f"{cls_name} has no registered name"
    assert _METRIC_REGISTRY[cls.name] is cls


@pytest.mark.parametrize("cls_name", NAMED_METRICS)
def test_get_metric_class_roundtrip(cls_name):
    cls = getattr(se, cls_name)
    assert get_metric_class(cls.name) is cls


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric_class("does_not_exist")


def test_names_are_unique():
    names = [getattr(se, c).name for c in NAMED_METRICS]
    assert len(names) == len(set(names))
