"""CompositeScore — normalized weighted blend of other metrics on the same node.

Reads ``node.metric_results`` for the named metrics; best used as a node's
``key_metric`` (run last). Unknown metrics are ignored, absent ones contribute 0.
"""

import pytest

from structured_eval.metrics import CharacterF1, CompositeScore, Fuzzy
from structured_eval.models import EvalConfig, EvalNode, FieldConfig
from tests.conftest import build_tree

pytestmark = pytest.mark.unit


def _field_node(config: EvalConfig) -> EvalNode:
    root = build_tree({"a": "kitten"}, {"a": "sitting"}, config)
    return next(c for c in root.children_nodes() if c.path == "a")


def test_weights_normalized_and_blended() -> None:
    """Half character_f1 + half fuzzy, weights normalized to sum 1.0."""
    config = EvalConfig(
        fields={
            "a": FieldConfig(
                metrics=[CharacterF1(), Fuzzy()],
                key_metric=CompositeScore({"character_f1": 1.0, "fuzzy": 1.0}),
            )
        }
    )
    node = _field_node(config)
    cf1 = float(node.metric_results["character_f1"])
    fz = float(node.metric_results["fuzzy"])
    assert float(node.metric_results["composite_score"]) == pytest.approx(
        0.5 * cf1 + 0.5 * fz
    )


def test_unknown_metrics_ignored() -> None:
    """A metric on the node but not in weights does not contribute."""
    config = EvalConfig(
        fields={
            "a": FieldConfig(
                metrics=[CharacterF1(), Fuzzy()],
                key_metric=CompositeScore({"character_f1": 1.0}),
            )
        }
    )
    node = _field_node(config)
    assert float(node.metric_results["composite_score"]) == pytest.approx(
        float(node.metric_results["character_f1"])
    )


def test_absent_metric_contributes_zero() -> None:
    """A weighted name not present on the node drops to 0 (still normalized)."""
    config = EvalConfig(
        fields={
            "a": FieldConfig(
                metrics=[CharacterF1()],
                key_metric=CompositeScore({"character_f1": 1.0, "fuzzy": 1.0}),
            )
        }
    )
    node = _field_node(config)
    cf1 = float(node.metric_results["character_f1"])
    assert float(node.metric_results["composite_score"]) == pytest.approx(0.5 * cf1)


@pytest.mark.parametrize(
    "weights",
    [{}, {"character_f1": 0.0}, {"a": -1.0, "b": 1.0}],
    ids=["empty", "zero-sum", "non-positive-sum"],
)
def test_invalid_weights_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        CompositeScore(weights)
