from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from structured_eval.model.result import RuleResult


class RuleProcessor:
    """Evaluates a list of business rules against a document.

    Each rule must expose ``evaluate(document) -> RuleResult`` (satisfied by both
    ``Rule`` and the result of ``Rule.custom()``). ``run`` returns the per-rule
    results and the pass rate (1.0 when there are no rules).
    """

    def run(self, rules: list[Any], document: dict[str, Any]) -> tuple[list[RuleResult], float]:
        results: list[RuleResult] = [rule.evaluate(document) for rule in rules]
        if not results:
            return results, 1.0
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        return results, pass_rate
