"""Verdicts for object metrics: matched fields → `(score, threshold, weight)`.

Concepts:

- TP (True Positive) — a `matched` entry, present on both sides;
- FP (False Positive) — a `spurious` entry, produced but not expected;
- FN (False Negative) — a `missing` entry, expected but not produced.

A parent object does not re-compare its children; it reads each matched child's
already-computed representative score and pairs it with the bar it must clear
and the weight it carries. Those triples feed `calculate.prf_counts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import resolve_metric
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.metrics.utils.calculate import WeightMode

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode
    from structured_eval.models.nodes.object_node import ObjectNode


def leaf_name(path: str) -> str:
    """Last path segment without any trailing index, e.g. `"a.b[0]"` → `"b"`."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _resolve_threshold(
    thresholds: float | dict[str, float] | None, name: str, fallback: float
) -> float:
    if isinstance(thresholds, dict):
        return float(thresholds.get(name, fallback))
    if thresholds is not None:
        return float(thresholds)
    return fallback


def _weight_of(child: EvalNode, weight_mode: WeightMode) -> float:
    return child.weight if weight_mode == WeightMode.PROPORTIONAL else 1.0


def matched_verdicts(
    node: ObjectNode,
    score_policy: dict[str, Any] | None = None,
    thresholds: float | dict[str, float] | None = None,
    weight_mode: WeightMode = WeightMode.PROPORTIONAL,
) -> list[tuple[float, float, float]]:
    """`(score, threshold, weight)` for each matched child of an object.

    Each child contributes its representative score — scalars and nested
    objects or arrays alike.

    Args:
        node: The object node whose matched children to grade.
        score_policy: Per-child metric override, re-scoring that child with the
            named metric whatever its kind.
        thresholds: One bar for every child, or a per-child dict.
        weight_mode: How each child's weight is derived.

    Returns:
        One verdict per matched child.
    """
    out: list[tuple[float, float, float]] = []
    for child in node.matched:
        name = leaf_name(child.path)
        spec = (score_policy or {}).get(name)
        if spec is not None:
            score = MetricInvoker(resolve_metric(spec)).scalar_on_node(child)
        else:
            score = child.representative
        threshold = _resolve_threshold(thresholds, name, child.threshold)
        out.append((score, threshold, _weight_of(child, weight_mode)))
    return out


def missing_weight(
    node: ObjectNode, weight_mode: WeightMode = WeightMode.PROPORTIONAL
) -> float:
    """Summed weight of the object's missing (FN) children (count when uniform)."""
    return sum(_weight_of(node.children[name], weight_mode) for name in node.missing)


def spurious_weight(
    node: ObjectNode, weight_mode: WeightMode = WeightMode.PROPORTIONAL
) -> float:
    """Summed weight of the object's spurious (FP) children (count when uniform)."""
    return sum(_weight_of(node.children[name], weight_mode) for name in node.spurious)
