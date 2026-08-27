"""Phase 3 of the pipeline: flattening the computed tree into a report."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from structured_eval.models import (
    ArrayNode,
    EvalReport,
    EvalWarning,
    FieldScore,
    MetricCollection,
    NodeType,
    ObjectNode,
    ScalarNode,
)

if TYPE_CHECKING:
    from structured_eval.models.context import EvalContext
    from structured_eval.models.nodes.base import EvalNode


class ReportBuilder:
    """Phase 3: flatten the computed node tree into an `EvalReport`.

    One walk of the tree produces every view the report offers: a `FieldScore`
    per path, a cross-field `MetricCollection` per metric name, and the
    alignment result of each array node. The headline `score` is the root node's
    representative (key-metric) value.

    Example:
        >>> from structured_eval.engine import (
        ...     MetricRunner, ReportBuilder, TreeBuilder)
        >>> from structured_eval.models import EvalConfig, EvalContext
        >>> from structured_eval.utils import flatten
        >>> actual, expected = {"status": "paid"}, {"status": "due"}
        >>> context = EvalContext(
        ...     actual=actual,
        ...     expected=expected,
        ...     source=None,
        ...     flat_actual=flatten(actual),
        ...     flat_expected=flatten(expected),
        ...     config=EvalConfig()
        ... )
        >>> root, warnings = TreeBuilder(context).build()
        >>> MetricRunner().run(root)
        >>> report = ReportBuilder().build(root, context, warnings)
        >>> report.score, report.score_label
        (0.0, 'mean_score')
        >>> sorted(report.field_scores)
        ['$', 'status']
    """

    _NODE_TYPE: ClassVar[dict[type, NodeType]] = {
        ScalarNode: NodeType.SCALAR,
        ObjectNode: NodeType.OBJECT,
        ArrayNode: NodeType.ARRAY,
    }

    def build(
        self, root: EvalNode, context: EvalContext, warnings: list[EvalWarning]
    ) -> EvalReport:
        """Flatten a tree whose metrics are already computed.

        Args:
            root: The root node, after `MetricRunner` has run.
            context: The context the tree was built from.
            warnings: What `TreeBuilder` collected while shaping the tree.

        Returns:
            The report: the document score and its label, one field score per
            path, the per-metric collections, the array alignments and the
            warnings.

        Example:
            >>> from structured_eval.engine import (
            ...     MetricRunner, ReportBuilder, TreeBuilder)
            >>> from structured_eval.models import EvalConfig, EvalContext
            >>> from structured_eval.utils import flatten
            >>> actual, expected = {"status": "paid"}, {"status": "due"}
            >>> context = EvalContext(
            ...     actual=actual,
            ...     expected=expected,
            ...     source=None,
            ...     flat_actual=flatten(actual),
            ...     flat_expected=flatten(expected),
            ...     config=EvalConfig()
            ... )
            >>> root, warnings = TreeBuilder(context).build()
            >>> MetricRunner().run(root)
            >>> report = ReportBuilder().build(root, context, warnings)
            >>> sorted(report.metrics)
            ['exact_match', 'mean_score', 'object_accuracy']
            >>> report.field_scores["status"].actual
            'paid'
        """
        field_scores = {}
        array_matches = {}
        # report.metrics is a cross-field view: each metric name → its value at
        # every node that produced it (a MetricCollection), built as we walk.
        # A metric's structured detail (schema errors, hallucinated paths, …)
        # rides along on each value's `.extra`.
        collections: dict[str, MetricCollection] = {}
        for node in root.walk():
            field_scores[node.path] = self._field_score(node)
            for name, result in node.metric_results.items():
                coll = collections.setdefault(name, MetricCollection(name=name))
                coll.by_path[node.path] = result
            if isinstance(node, ArrayNode) and node.match_result is not None:
                array_matches[node.path] = node.match_result

        # The headline number is the root node's representative (key) metric.
        score_label = root.key_metric.name if root.key_metric is not None else None
        root_score = (
            root.metric_results.get(score_label) if score_label is not None else None
        )
        score = float(root_score) if root_score is not None else None

        return EvalReport(
            score=score,
            score_label=score_label,
            metrics=collections,
            field_scores=field_scores,
            array_matches=array_matches,
            warnings=warnings,
        )

    def _field_score(self, node: EvalNode) -> FieldScore:
        """One node as the report's per-field entry."""
        return FieldScore(
            path=node.path,
            node_type=self._NODE_TYPE.get(type(node), NodeType.SCALAR),
            actual=node.actual,
            expected=node.expected,
            metrics=dict(node.metric_results),
            score=node.representative,  # the node's representative (key-metric) value
            threshold=node.threshold,
        )
