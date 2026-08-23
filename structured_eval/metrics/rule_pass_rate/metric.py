"""The `rule_pass_rate` metric — the fraction of rules a document satisfies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.rule_pass_rate.engine import RuleProcessor

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class RulePassRate(RootMetric):
    """Fraction of business rules that hold for the document.

    Per-rule outcomes ride on the result's `extra["rule_results"]`.
    Read them back via
    `report.metrics["rule_pass_rate"].extra_values("rule_results")`.

    An empty rule list scores 1.0, vacuously.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import Rule, RulePassRate
        >>> from structured_eval.models import EvalConfig
        >>> rules = [Rule("$.total").gt(0), Rule("$.total").lt(100)]
        >>> report = evaluate({"total": 250}, None,
        ...                   EvalConfig(metrics=[RulePassRate(rules)]))
        >>> float(report.metrics["rule_pass_rate"].representative())
        0.5
    """

    name = "rule_pass_rate"

    def __init__(self, rules: list[Any], name: str | None = None):
        """Bind the rules this metric checks.

        Args:
            rules: `Rule` or `Rule.custom(...)` objects, each exposing
                `evaluate(document) -> RuleResult`.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.rules = rules
        self.processor = RuleProcessor()

    def compute(self, node: EvalNode) -> tuple[float, dict[str, Any]]:
        """The share of rules that pass, plus every rule's own outcome."""
        document = node.actual
        results, pass_rate = self.processor.run(
            self.rules, document if isinstance(document, dict) else {}
        )
        return pass_rate, {"rule_results": [r.model_dump() for r in results]}
