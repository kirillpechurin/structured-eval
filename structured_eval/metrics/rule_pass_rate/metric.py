from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.rule_pass_rate.engine import RuleProcessor
from structured_eval.model.nodes.base import EvalNode


class RulePassRate(RootMetric):
    """Fraction of business rules that hold for the document.

    ``rules`` is a list of ``Rule`` (DSL) or ``Rule.custom(...)`` objects, each
    exposing ``evaluate(document) -> RuleResult``. Per-rule outcomes are returned
    as the result's ``extra["rule_results"]`` (serialized ``RuleResult`` dicts) —
    read via ``report.metrics["rule_pass_rate"].extra_values("rule_results")``. An
    empty rule list scores 1.0 (vacuously true).
    """

    name = "rule_pass_rate"

    def __init__(self, rules: list[Any]):
        self.rules = rules
        self.processor = RuleProcessor()

    def compute(self, node: EvalNode) -> tuple[float, dict[str, Any]]:
        document = node.actual
        results, pass_rate = self.processor.run(
            self.rules, document if isinstance(document, dict) else {}
        )
        return pass_rate, {"rule_results": [r.model_dump() for r in results]}
