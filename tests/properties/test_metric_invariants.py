"""Case A — field-metric invariants over generated scalar pairs.

A field metric is a pure function ``score(actual, expected) -> float``. Whatever
the LLM emits, that function must obey four laws. Examples can't prove a law;
these sweep a seeded corpus so a single counterexample fails the build with a
reproducible seed.

  * Boundedness   — result is always a finite float in [0, 1]; never raises.
  * Identity      — ``score(x, x) == 1.0`` on the metric's own domain.
  * Symmetry      — order-independent metrics give ``score(a,e) == score(e,a)``.
  * String-only   — text metrics return exactly 0.0 on any non-str input.
"""

import math
import random
from typing import Any

import pytest

from structured_eval import (
    ExactMatch,
    Fuzzy,
    Levenshtein,
    Numeric,
    NumericCloseness,
    RegexMatch,
    TokenF1,
    TypeMatch,
)

from .conftest import SEEDS, random_scalar, random_str

pytestmark = pytest.mark.property

# Every value metric that exposes the pure ``score(actual, expected)`` primitive.
# (Presence/FieldFaithfulness override compute(node) instead and are covered at
# the engine layer.)
ALL_SCORE_METRICS = [
    ExactMatch(),
    TypeMatch(),
    RegexMatch(),
    TokenF1(),
    Fuzzy(),
    Levenshtein(),
    Numeric(),
    NumericCloseness(),
]

STRING_ONLY = [RegexMatch(), TokenF1(), Fuzzy(), Levenshtein()]
SYMMETRIC = [ExactMatch(), TypeMatch(), Fuzzy(), TokenF1(), NumericCloseness()]


def _ids(metrics: list[Any]) -> list[str]:
    return [type(m).__name__ for m in metrics]


@pytest.mark.parametrize("metric", ALL_SCORE_METRICS, ids=_ids(ALL_SCORE_METRICS))
@pytest.mark.parametrize("seed", SEEDS)
def test_boundedness(metric: Any, seed: Any) -> None:
    """Any scalar pair → a finite float in [0, 1], with no exception escaping."""
    rng = random.Random(seed)
    for _ in range(20):
        a, e = random_scalar(rng), random_scalar(rng)
        score = metric.score(a, e)
        assert isinstance(score, float)
        assert math.isfinite(score), f"{type(metric).__name__}({a!r},{e!r})={score}"
        assert 0.0 <= score <= 1.0, f"{type(metric).__name__}({a!r},{e!r})={score}"


@pytest.mark.parametrize("seed", SEEDS)
def test_identity_universal(seed: Any) -> None:
    """ExactMatch/TypeMatch are reflexive for every scalar (incl. None)."""
    rng = random.Random(seed)
    for _ in range(20):
        x = random_scalar(rng)
        assert ExactMatch().score(x, x) == 1.0
        assert TypeMatch().score(x, x) == 1.0


@pytest.mark.parametrize(
    "metric", [RegexMatch(), TokenF1(), Fuzzy(), Levenshtein()], ids=_ids(STRING_ONLY)
)
@pytest.mark.parametrize("seed", SEEDS)
def test_identity_string_domain(metric: Any, seed: Any) -> None:
    """Text metrics are reflexive on their own domain — equal strings → 1.0."""
    rng = random.Random(seed)
    for _ in range(20):
        s = random_str(rng)
        assert metric.score(s, s) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", SYMMETRIC, ids=_ids(SYMMETRIC))
@pytest.mark.parametrize("seed", SEEDS)
def test_symmetry(metric: Any, seed: Any) -> None:
    """Order-independent metrics: swapping operands never changes the score."""
    rng = random.Random(seed)
    for _ in range(20):
        a, e = random_scalar(rng), random_scalar(rng)
        assert metric.score(a, e) == pytest.approx(metric.score(e, a))


@pytest.mark.parametrize("metric", STRING_ONLY, ids=_ids(STRING_ONLY))
@pytest.mark.parametrize("seed", SEEDS)
def test_string_only_contract(metric: Any, seed: Any) -> None:
    """A non-str on either side scores exactly 0.0 — never coerced, never raised."""
    rng = random.Random(seed)
    for _ in range(20):
        s = random_str(rng)
        non_str = rng.choice([1, 1.5, True, None, ["x"], {"k": 1}])
        assert metric.score(non_str, s) == 0.0
        assert metric.score(s, non_str) == 0.0
