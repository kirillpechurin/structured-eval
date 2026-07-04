"""Cases C & D — metamorphic relations across related documents.

Metamorphic tests assert how the score *changes* when the input changes in a
known direction — stronger than pinning a single number, and immune to formula
re-tuning that preserves ordering.

  * C Monotonic degradation — corrupting one more field never raises the score.
  * D Weight decomposition  — with uniform weights, PROPORTIONAL collapses to
    the plain count-based (NONE) result, exactly as CLAUDE.md promises.
"""

import random
from typing import Any

import pytest

from structured_eval import evaluate
from structured_eval.metrics import ObjectAccuracy, ObjectF1, OverallLeafScore
from structured_eval.metrics.utils.calculate import WeightMode
from structured_eval.models import EvalConfig

from .conftest import SEEDS

pytestmark = pytest.mark.property


@pytest.mark.parametrize("seed", SEEDS)
def test_corrupting_more_fields_never_raises_score(seed: Any) -> None:
    """Start from a perfect doc; corrupt leaves one at a time → score is monotone non-increasing."""
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    expected = {f"f{i}": rng.randint(0, 100) for i in range(n)}

    cfg = EvalConfig(metrics=[OverallLeafScore()])
    keys = list(expected)
    rng.shuffle(keys)

    actual = dict(expected)
    prev = (
        evaluate(actual, expected, config=cfg)
        .metrics["overall_leaf_score"]
        .representative()
    )
    assert prev == pytest.approx(1.0)

    for k in keys:
        actual = dict(actual)
        actual[k] = expected[k] + 10_000  # guaranteed-wrong value
        cur = (
            evaluate(actual, expected, config=cfg)
            .metrics["overall_leaf_score"]
            .representative()
        )
        assert cur <= prev + 1e-9, f"score rose after corrupting {k}: {prev} -> {cur}"
        prev = cur

    assert prev == pytest.approx(0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_uniform_weights_reduce_proportional_to_counts(seed: Any) -> None:
    """With equal child weights, PROPORTIONAL F1 == NONE (count-based) F1."""
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    expected = {f"f{i}": rng.randint(0, 100) for i in range(n)}
    # Corrupt a random subset so the score is non-trivial.
    actual = {k: (v + 1 if rng.random() < 0.5 else v) for k, v in expected.items()}

    prop = (
        evaluate(
            actual,
            expected,
            config=EvalConfig(metrics=[ObjectF1(weight_mode=WeightMode.PROPORTIONAL)]),
        )
        .metrics["object_f1"]
        .representative()
    )
    none = (
        evaluate(
            actual,
            expected,
            config=EvalConfig(metrics=[ObjectF1(weight_mode=WeightMode.NONE)]),
        )
        .metrics["object_f1"]
        .representative()
    )

    assert prop == pytest.approx(none)


@pytest.mark.parametrize("seed", SEEDS)
def test_accuracy_equals_fraction_correct(seed: Any) -> None:
    """ObjectAccuracy over flat scalar fields == fraction of exactly-correct fields."""
    rng = random.Random(seed)
    n = rng.randint(1, 10)
    expected = {f"f{i}": rng.randint(0, 9) for i in range(n)}
    actual = {k: (v if rng.random() < 0.6 else v + 1) for k, v in expected.items()}
    correct = sum(1 for k in expected if actual[k] == expected[k])

    acc = (
        evaluate(
            actual,
            expected,
            config=EvalConfig(metrics=[ObjectAccuracy(weight_mode=WeightMode.NONE)]),
        )
        .metrics["object_accuracy"]
        .representative()
    )
    assert acc == pytest.approx(correct / n)
