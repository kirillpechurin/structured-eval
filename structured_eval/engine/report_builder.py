from __future__ import annotations

from structured_eval.metrics._shared import match_criterion as mc
from structured_eval.model.context import EvalContext
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode
from structured_eval.model.result import EvalReport, FieldScore, NodeType, RuleResult


class ReportBuilder:
    """Phase 3: flatten the computed node tree into an ``EvalReport``."""

    _NODE_TYPE = {
        ScalarNode: NodeType.SCALAR,
        ObjectNode: NodeType.OBJECT,
        ArrayNode: NodeType.ARRAY,
    }

    def build(self, root: EvalNode, context: EvalContext, warnings: list[str]) -> EvalReport:
        field_scores = {}
        array_matches = {}
        for node in root.walk():
            field_scores[node.path] = self._field_score(node)
            if isinstance(node, ArrayNode) and node.match_result is not None:
                array_matches[node.path] = node.match_result

        metrics = dict(root.metric_results)  # root carries document-level metrics
        config = context.config
        # The headline number is the root node's representative (key) metric.
        score_label = root.key_metric.name if root.key_metric is not None else None
        score = metrics.get(score_label) if score_label is not None else None

        # Metrics expose extra detail as side channels; collect it uniformly.
        schema_errors: list[str] = []
        rule_results: list[RuleResult] = []
        hallucinated_fields: list[str] = []
        for metric in [*config.metrics, config.key_metric]:
            schema_errors.extend(getattr(metric, "schema_errors", []) or [])
            rule_results.extend(getattr(metric, "rule_results", []) or [])
            hallucinated_fields.extend(getattr(metric, "hallucinated_fields", []) or [])

        return EvalReport(
            score=score,
            score_label=score_label,
            metrics=metrics,
            field_scores=field_scores,
            array_matches=array_matches,
            rule_results=rule_results,
            schema_errors=schema_errors,
            hallucinated_fields=hallucinated_fields,
            warnings=warnings,
        )

    def _field_score(self, node: EvalNode) -> FieldScore:
        return FieldScore(
            path=node.path,
            node_type=self._NODE_TYPE.get(type(node), NodeType.SCALAR),
            actual=node.actual,
            expected=node.expected,
            metrics=dict(node.metric_results),
            score=mc.repr_score(node),  # the node's representative (key-metric) value
            threshold=node.threshold,
        )
