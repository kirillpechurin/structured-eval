"""Contracts every metric must satisfy — driven by the registry.

These run once over *all* registered metrics, so a newly added metric inherits
the baseline guarantees for free. Per-metric files assert behaviour; this file
asserts the laws shared by the whole family. Do not re-assert these per metric.
"""

from typing import Any

import pytest

from structured_eval.metrics.base import (
    _METRIC_REGISTRY,
    BaseMetric,
    FieldMetric,
    resolve_metric,
)

pytestmark = pytest.mark.unit

ALL_METRICS = sorted(_METRIC_REGISTRY.items())


def _instantiable_field_score_metrics() -> list[tuple[str, FieldMetric]]:
    """Field metrics with a no-arg ctor and a pure ``score`` (engine-free).

    Excludes node-based field metrics (Presence inspects the node;
    FieldFaithfulness requires a source) and anything needing constructor args.
    """
    out = []
    for name, cls in ALL_METRICS:
        if not issubclass(cls, FieldMetric):
            continue
        if name in {"presence", "field_faithfulness"}:
            continue
        try:
            out.append((name, cls()))
        except TypeError:
            continue  # needs constructor args → covered in its own file
    return out


FIELD_SCORE_METRICS = _instantiable_field_score_metrics()
_FIELD_IDS = [name for name, _ in FIELD_SCORE_METRICS]
_FIELD_INSTANCES = [m for _, m in FIELD_SCORE_METRICS]

# Deliberately messy inputs an LLM may emit — a metric must stay total on these.
GARBAGE_PAIRS = [
    (None, None),
    ("x", 1),
    (1, "x"),
    (True, False),
    ([1, 2], {"a": 1}),
    ("", ""),
    (1.5, -3),
    ({"k": "v"}, None),
]


# ── registry integrity ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("name", "cls"), ALL_METRICS, ids=[n for n, _ in ALL_METRICS])
def test_registered_under_a_nonempty_name(name: Any, cls: Any) -> None:
    assert isinstance(name, str) and name
    assert issubclass(cls, BaseMetric)
    assert cls.name == name


# Metrics with required constructor args can't be resolved from a bare name.
NO_ARG_METRICS = []
for _name, _cls in ALL_METRICS:
    try:
        _cls()
    except TypeError:
        continue
    NO_ARG_METRICS.append((_name, _cls))


@pytest.mark.parametrize(
    ("name", "cls"), NO_ARG_METRICS, ids=[n for n, _ in NO_ARG_METRICS]
)
def test_resolve_by_name_returns_an_instance(name: Any, cls: Any) -> None:
    assert isinstance(resolve_metric(name), cls)


def test_resolve_passes_instances_through() -> None:
    inst = next(m for _, m in FIELD_SCORE_METRICS)
    assert resolve_metric(inst) is inst


# ── score laws (field metrics) ───────────────────────────────────────────────


@pytest.mark.parametrize("metric", _FIELD_INSTANCES, ids=_FIELD_IDS)
@pytest.mark.parametrize(
    "pair", GARBAGE_PAIRS, ids=[f"{i}" for i in range(len(GARBAGE_PAIRS))]
)
def test_score_is_total_and_bounded(metric: Any, pair: Any) -> None:
    """Never raises, and always returns a float in [0, 1]."""
    actual, expected = pair
    score = metric.score(actual, expected)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"{metric.name}({actual!r},{expected!r}) = {score}"
