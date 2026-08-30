"""MetricResult — a float everywhere, with structured detail attached.

Being a `float` subclass is the point: every consumer that averages, compares
or serializes a score keeps working, and the metric that has something to say
attaches it to `.extra` without a wrapper anyone has to unwrap.

What is pinned here is that both halves survive: arithmetic behaves like a
float, and `extra` round-trips through pydantic.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from structured_eval.models import MetricResult

pytestmark = pytest.mark.unit


class Holder(BaseModel):
    """A model with a `MetricResult` field — the embedding under test."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: MetricResult


def test_it_is_a_float() -> None:
    result = MetricResult(0.5)
    assert result == 0.5
    assert result + 0.5 == 1.0
    assert float(result) == 0.5


def test_extra_defaults_to_empty() -> None:
    assert MetricResult(1.0).extra == {}


def test_extra_is_copied_not_aliased() -> None:
    # The caller's dict must not become the result's mutable state.
    source: dict[str, Any] = {"reason": "invented"}
    result = MetricResult(0.0, source)
    source["reason"] = "changed"
    assert result.extra == {"reason": "invented"}


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (MetricResult(0.5), "MetricResult(0.5)"),
        (MetricResult(0.0, {"why": "x"}), "MetricResult(0.0, extra={'why': 'x'})"),
    ],
    ids=["bare", "with-extra"],
)
def test_repr_shows_extra_only_when_there_is_some(
    result: MetricResult, expected: str
) -> None:
    assert repr(result) == expected


@pytest.mark.parametrize(
    ("result", "dumped"),
    [
        # Nothing to say → nothing in the report but the number.
        (MetricResult(0.5), 0.5),
        (MetricResult(0.0, {"why": "x"}), {"value": 0.0, "extra": {"why": "x"}}),
    ],
    ids=["bare", "with-extra"],
)
def test_serialization_keeps_the_number_bare_unless_there_is_detail(
    result: MetricResult, dumped: Any
) -> None:
    assert Holder(value=result).model_dump() == {"value": dumped}


@pytest.mark.parametrize(
    "raw",
    [0.5, {"value": 0.5}, {"value": 0.5, "extra": None}],
    ids=["bare-float", "mapping", "mapping-null-extra"],
)
def test_both_serialized_forms_re_validate(raw: Any) -> None:
    holder = Holder.model_validate({"value": raw})
    assert holder.value == 0.5
    assert holder.value.extra == {}


def test_extra_survives_the_round_trip() -> None:
    original = Holder(value=MetricResult(0.0, {"why": "x"}))
    assert Holder.model_validate(original.model_dump()).value.extra == {"why": "x"}


def test_validating_an_existing_result_keeps_it() -> None:
    result = MetricResult(0.25, {"why": "x"})
    assert Holder.model_validate({"value": result}).value is result
