"""Case B — alignment optimality: Hungarian dominates greedy ByKey.

CLAUDE.md states the relationship explicitly: ``HungarianAligner`` is the
*optimal* one-to-one assignment, while ``ByKeyAligner`` is a cheap
*globally-greedy* approximation. The optimum can never match fewer pairs than
the approximation. We pin that as:

  * a property — over random key lists, ``len(hungarian.matched) >=
    len(bykey.matched)`` with identical exact-match scoring;
  * a non-vacuous baseline — on a recoverable permutation both reach the
    ceiling (the >= is not satisfied by one strategy matching nothing); and
  * score-level dominance — on a graded cross-match Hungarian's ArrayF1 is
    never below greedy's.
"""

import random

import pytest

from structured_eval import (
    ArrayF1,
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExactMatch,
    FieldConfig,
    NumericCloseness,
    ObjectFieldConfig,
    evaluate,
)
from structured_eval.alignment import ByKeyAligner, HungarianAligner

from .conftest import SEEDS

pytestmark = pytest.mark.property


@pytest.mark.parametrize("seed", SEEDS)
def test_hungarian_matches_at_least_as_many_as_greedy(seed) -> None:
    """Optimal assignment matches >= greedy, given the same exact-key scoring."""
    rng = random.Random(seed)
    pool = list("abcde")
    expected = [{"id": rng.choice(pool)} for _ in range(rng.randint(0, 6))]
    actual = [{"id": rng.choice(pool)} for _ in range(rng.randint(0, 6))]

    greedy = ByKeyAligner(key="id", threshold=1.0).align(expected, actual)
    optimal = HungarianAligner(scorer=ExactMatch(), threshold=1.0, key="id").align(expected, actual)

    assert len(optimal.matched) >= len(greedy.matched)
    # And the optimum can't exceed the obvious ceiling of available pairs.
    assert len(optimal.matched) <= min(len(expected), len(actual))


def test_both_strategies_recover_a_reordered_list() -> None:
    """On a clean permutation both strategies find the full one-to-one matching.

    This pins the non-vacuous baseline: the inequality above is not trivially
    satisfied by one strategy always matching nothing — given a recoverable
    problem, greedy and optimal both reach the ceiling.
    """
    expected = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    actual = [{"id": "c"}, {"id": "a"}, {"id": "b"}]

    greedy = ByKeyAligner(key="id", threshold=1.0).align(expected, actual)
    optimal = HungarianAligner(scorer=ExactMatch(), threshold=1.0, key="id").align(expected, actual)

    assert len(greedy.matched) == 3
    assert len(optimal.matched) == 3
    assert optimal.missed == [] and optimal.spurious == []


def test_hungarian_total_similarity_dominates_on_crossmatch() -> None:
    """At the *score* level Hungarian never scores below greedy on a cross-match.

    Optimal assignment maximises total similarity, so the value-aware ArrayF1
    under HUNGARIAN is >= the same metric under the greedy BY_KEY strategy.
    """
    expected = [{"v": 10.0}, {"v": 20.0}]
    actual = [{"v": 19.0}, {"v": 11.0}]  # swapped near-matches

    item = ObjectFieldConfig(fields={"v": FieldConfig(metrics=[NumericCloseness()])})
    greedy_cfg = EvalConfig(
        metrics=[ArrayF1()],
        fields={
            "xs": ArrayFieldConfig(
                strategy=ArrayStrategy.BY_KEY,
                params={"key": "v", "key_metric": NumericCloseness(), "threshold": 0.5},
                item=item,
            )
        },
    )
    opt_cfg = EvalConfig(
        metrics=[ArrayF1()],
        fields={
            "xs": ArrayFieldConfig(
                strategy=ArrayStrategy.HUNGARIAN,
                params={"scorer": {"v": NumericCloseness()}, "threshold": 0.5},
                item=item,
            )
        },
    )

    g = evaluate({"xs": actual}, {"xs": expected}, config=greedy_cfg)
    o = evaluate({"xs": actual}, {"xs": expected}, config=opt_cfg)
    assert o.field_scores["xs"].metrics["array_f1"] >= g.field_scores["xs"].metrics["array_f1"]
