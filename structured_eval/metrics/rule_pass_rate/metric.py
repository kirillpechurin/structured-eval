from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.rule_pass_rate.engine import RuleProcessor
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.result import RuleResult


class RulePassRate(RootMetric):
    """Fraction of business rules that hold for the document.

    ``rules`` is a list of ``Rule`` (DSL) or ``Rule.custom(...)`` objects, each
    exposing ``evaluate(document) -> RuleResult``. Per-rule outcomes are kept in
    ``self.rule_results`` and surfaced into ``report.rule_results``. An empty
    rule list scores 1.0 (vacuously true).
    """

    name = "rule_pass_rate"

    def __init__(self, rules: list[Any]):
        self.rules = rules
        self.processor = RuleProcessor()
        self.rule_results: list[RuleResult] = []

    def compute(self, node: EvalNode) -> float:
        document = node.actual
        results, pass_rate = self.processor.run(
            self.rules, document if isinstance(document, dict) else {}
        )
        self.rule_results = results
        return pass_rate
