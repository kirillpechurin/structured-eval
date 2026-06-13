from __future__ import annotations

from structured_eval.core.context import EvalContext
from structured_eval.core.result import EvalReport, FieldScore
from structured_eval.engine.compute import walk
from structured_eval.metrics import _match_criterion as mc
from structured_eval.nodes.array_node import ArrayNode
from structured_eval.nodes.base import EvalNode
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode

_NODE_TYPE = {ScalarNode: "scalar", ObjectNode: "object", ArrayNode: "array"}


def _field_score(node: EvalNode) -> FieldScore:
    score: float | None = None
    threshold: float | None = None
    if isinstance(node, ScalarNode):
        score, threshold = mc.field_verdict(node)  # key (match-criterion) value
    return FieldScore(
        path=node.path,
        node_type=_NODE_TYPE.get(type(node), "scalar"),
        actual=node.actual,
        expected=node.expected,
        metrics=dict(node.metric_results),
        score=score,
        threshold=threshold,
    )


def build_report(root: EvalNode, context: EvalContext, warnings: list[str]) -> EvalReport:
    """Phase 3: flatten the computed tree into an EvalReport."""
    field_scores = {}
    array_matches = {}
    for node in walk(root):
        field_scores[node.path] = _field_score(node)
        if isinstance(node, ArrayNode) and node.match_result is not None:
            array_matches[node.path] = node.match_result

    metrics = dict(root.metric_results)  # root node carries the document-level metrics
    config = context.config
    score: float | None = None
    score_label: str | None = None
    if config.key_metric is not None:
        score_label = config.key_metric.name
        score = metrics.get(score_label)

    return EvalReport(
        score=score,
        score_label=score_label,
        metrics=metrics,
        field_scores=field_scores,
        array_matches=array_matches,
        warnings=warnings,
    )
