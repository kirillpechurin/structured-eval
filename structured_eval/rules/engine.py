from __future__ import annotations

from typing import Any

from structured_eval.core.result import RuleResult


def run_rules(
    rules: list[Any],
    document: dict[str, Any],
) -> tuple[list[RuleResult], float]:
    """Evaluate a list of rules against a document.

    Each rule must expose an ``evaluate(document) -> RuleResult`` method
    (satisfied by both ``Rule`` and the result of ``Rule.custom()``).

    Returns:
        A tuple of (results, pass_rate). pass_rate is 1.0 when rules is empty.
    """
    results: list[RuleResult] = [rule.evaluate(document) for rule in rules]
    if not results:
        return results, 1.0
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    return results, pass_rate
