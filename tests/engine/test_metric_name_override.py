"""Per-instance metric names: two configurations of one metric on one node.

Without a custom name both instances key on the class name and the second
overwrites the first in ``node.metric_results`` / ``report.metrics``.
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval.metrics import Numeric
from structured_eval.models import EvalConfig, EvalReport, FieldConfig

pytestmark = pytest.mark.engine

ACTUAL = {"total": 100.5}
EXPECTED = {"total": 100.0}


def test_two_configs_of_one_metric_coexist(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """0.5% deviation: outside a 0.1% band, inside a 10% one."""
    config = EvalConfig(
        fields={
            "total": FieldConfig(
                metrics=[
                    Numeric(tolerance=0.001, name="strict"),
                    Numeric(tolerance=0.1, name="loose"),
                ]
            )
        }
    )
    report = evaluate_one(ACTUAL, EXPECTED, config)

    assert report.metrics["strict"].by_path["total"] == 0.0
    assert report.metrics["loose"].by_path["total"] == 1.0
    assert "numeric" not in report.metrics
    assert report.field_scores["total"].metrics.keys() >= {"strict", "loose"}


def test_without_custom_names_the_second_instance_wins(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """The collision this feature exists to remove — pinned as current behaviour."""
    config = EvalConfig(
        fields={
            "total": FieldConfig(
                metrics=[Numeric(tolerance=0.001), Numeric(tolerance=0.1)]
            )
        }
    )
    report = evaluate_one(ACTUAL, EXPECTED, config)

    assert set(report.field_scores["total"].metrics) == {"numeric", "mean_score"}
    assert report.metrics["numeric"].by_path["total"] == 1.0  # the loose one


def test_default_name_is_used_when_none_given(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    config = EvalConfig(fields={"total": FieldConfig(metrics=[Numeric(tolerance=0.1)])})
    report = evaluate_one(ACTUAL, EXPECTED, config)
    assert report.metrics["numeric"].by_path["total"] == 1.0


def test_key_metric_resolves_against_a_custom_name(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """A custom name is a valid ``key_metric`` string — the instance is reused."""
    config = EvalConfig(
        fields={
            "total": FieldConfig(
                metrics=[
                    Numeric(tolerance=0.001, name="strict"),
                    Numeric(tolerance=0.1, name="loose"),
                ],
                key_metric="strict",
            )
        }
    )
    report = evaluate_one(ACTUAL, EXPECTED, config)

    field = report.field_scores["total"]
    assert field.score == 0.0  # representative = the strict instance
    assert "mean_score" not in field.metrics


def test_same_instance_config_is_not_deduplicated_by_name(
    tree_factory: Callable[..., Any],
) -> None:
    """Cascade dedup is by identity, so distinct instances both survive."""
    config = EvalConfig(
        metrics=[Numeric(tolerance=0.001, name="strict"), Numeric(0.1, name="loose")]
    )
    root = tree_factory(ACTUAL, EXPECTED, config)
    assert set(root.children["total"].metric_results) == {
        "strict",
        "loose",
        "mean_score",
    }
