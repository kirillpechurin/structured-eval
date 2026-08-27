"""Checking a document against a list of business rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from structured_eval.models.result import RuleResult


class RuleProcessor:
    """The `rule_pass_rate` metric's rule loop, over one document.

    A rule is anything exposing `evaluate(document) -> RuleResult`. `Rule` and
    what `Rule.custom()` returns both qualify, and so does a caller's own
    class: the parameter stays `list[Any]` so that stays true.
    """

    def run(
        self, rules: list[Any], document: dict[str, Any]
    ) -> tuple[list[RuleResult], float]:
        """Evaluate every rule against the document.

        Args:
            rules: The rules to run, in order.
            document: The document to check them against.

        Returns:
            The per-rule results and the pass rate, 1.0 when there are no rules.
        """
        results: list[RuleResult] = [rule.evaluate(document) for rule in rules]
        if not results:
            return results, 1.0
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        return results, pass_rate
