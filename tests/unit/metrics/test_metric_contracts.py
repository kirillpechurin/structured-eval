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
    assert isinstance(name, str)
    assert name
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


# ── per-instance name override ───────────────────────────────────────────────

# Constructor args for the metrics that require them. Exhaustiveness is asserted
# below, so a new arg-taking metric cannot silently skip the name contract.
ARG_METRICS: dict[str, tuple[Any, ...]] = {
    "composite_score": ({"exact_match": 1.0},),
    "rule_pass_rate": ([],),
    "schema_validity": ({"type": "object"},),
}

CONSTRUCTIBLE = [(name, cls, ARG_METRICS.get(name, ())) for name, cls in ALL_METRICS]
_CTOR_IDS = [name for name, _, _ in CONSTRUCTIBLE]


def test_arg_metric_table_is_exhaustive() -> None:
    """New metrics with required ctor args must be added to ``ARG_METRICS``."""
    needs_args = {n for n, _ in ALL_METRICS} - {n for n, _ in NO_ARG_METRICS}
    assert needs_args == set(ARG_METRICS)


@pytest.mark.parametrize(("name", "cls", "args"), CONSTRUCTIBLE, ids=_CTOR_IDS)
def test_omitting_name_keeps_the_class_name(name: Any, cls: Any, args: Any) -> None:
    assert cls(*args).name == name


@pytest.mark.parametrize(("name", "cls", "args"), CONSTRUCTIBLE, ids=_CTOR_IDS)
def test_custom_name_shadows_only_the_instance(name: Any, cls: Any, args: Any) -> None:
    """Every metric forwards ``name`` to BaseMetric; the class stays untouched.

    Guards the ``super().__init__(name=name)`` convention: a metric that defines
    ``__init__`` and forgets to forward will fail here.
    """
    instance = cls(*args, name="custom")
    assert instance.name == "custom"
    assert cls.name == name  # class attribute unchanged
    assert cls(*args).name == name  # a sibling instance is unaffected


@pytest.mark.parametrize(("name", "cls", "args"), CONSTRUCTIBLE, ids=_CTOR_IDS)
def test_registry_key_is_unaffected_by_a_custom_name(
    name: Any, cls: Any, args: Any
) -> None:
    cls(*args, name="custom")
    assert _METRIC_REGISTRY[name] is cls


@pytest.mark.parametrize(
    ("name", "cls"), NO_ARG_METRICS, ids=[n for n, _ in NO_ARG_METRICS]
)
def test_name_resolution_survives_a_custom_name(name: Any, cls: Any) -> None:
    """``resolve_metric("numeric")`` still yields a default-named instance."""
    cls(name="custom")
    assert resolve_metric(name).name == name


@pytest.mark.parametrize("bad", ["", None])
def test_name_must_be_a_nonempty_string(bad: Any) -> None:
    from structured_eval.metrics.exact import ExactMatch

    if bad is None:  # None means "no override" — the default path, not an error
        assert ExactMatch(name=bad).name == "exact_match"
    else:
        with pytest.raises(ValueError, match="non-empty"):
            ExactMatch(name=bad)


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
