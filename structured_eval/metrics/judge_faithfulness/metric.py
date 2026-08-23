"""JudgeFaithfulness — an LLM judge that grades a subtree, field by field."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.llm import resolve_client
from structured_eval.metrics.base import AnyNodeMetric
from structured_eval.metrics.judge_faithfulness.criteria import Criteria, relative_path
from structured_eval.metrics.judge_faithfulness.prompt import (
    DEFAULT_CRITERION,
    SYSTEM_PROMPT,
    build_prompt,
)
from structured_eval.metrics.judge_faithfulness.schemas import JudgeReply, Verdict
from structured_eval.models import FieldJudgeVerdict, JudgeVerdict, MetricResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from structured_eval.llm import LlmClient
    from structured_eval.models.nodes.base import EvalNode

# What each verdict is worth.
# - `supported` — 1.0: the source backs the value.
# - `contradicted` — 0.0: the source states something incompatible with it.
# - `not_stated` — 0.0: the source says nothing about it. A value the source
#   never states is exactly the fabrication this metric exists to catch, so it
#   cannot count as faithful.
DEFAULT_VERDICT_SCORES: dict[str, float] = {
    Verdict.SUPPORTED: 1.0,
    Verdict.NOT_STATED: 0.0,
    Verdict.CONTRADICTED: 0.0,
}


class JudgeFaithfulness(AnyNodeMetric):
    """Check that every leaf beneath the node is grounded in the `source`.

    It can be attached to any node.
    It takes every leaf beneath that node and asks one LLM call whether the
    source backs each value, ruling:

    - `supported`;
    - `not_stated`;
    - `contradicted`.

    Each verdict is scored through `verdict_scores`, and the node scores their
    weighted mean.

    The score lands on the judged node, and the judge's full result rides along
    in its `extra` as one `JudgeVerdict`.
    Read them back via
    `report.metrics["judge_faithfulness"].extra_values("verdict")`.

    Where to hang the judge is a real choice:

    - On a subtree — one call, the detail as evidence. A verdict does *not*
      become the score of the field it names.
    - On a single field — one call, the detail as that field's score.

    Only what the output actually produced is judged, never the ground truth:

    - An explicit `null` is itself a claim, so it gets a verdict.
    - A field absent from the output claims nothing, so it is left out.
      Missing it is a recall failure, which `Presence` and `CoverageLeafScore`
      already measure.

    Costs money, takes time and is not deterministic, so it runs only where it
    is configured. Needs a grounding `source`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import JudgeFaithfulness
        >>> from structured_eval.models import EvalConfig
        >>> judge = JudgeFaithfulness({"vendor": "must name the issuing company"})
        >>> report = evaluate(  # doctest: +SKIP
        ...     {"vendor": "Acme"}, None, EvalConfig(metrics=[judge]),
        ...     source="Invoice from Acme Ltd.",
        ... )
        >>> float(report.metrics["judge_faithfulness"].representative())  # doctest: +SKIP
        1.0
    """

    name = "judge_faithfulness"

    def __init__(
        self,
        criteria: str | dict[str, str] | None = None,
        *,
        client: Any = None,
        verdict_scores: dict[str, float] | None = None,
        name: str | None = None,
    ) -> None:
        """Configure what the judge is asked and which model answers.

        Args:
            criteria: What "faithful" means for the judged node, in one of
                three shapes:

                - one rule covering every field;
                - a rule per field, keyed by a path relative to the judged node;
                - `None` uses a generic formulation.

            client: The LLM client.
                If unset, it comes from the `STRUCTURED_EVAL_LLM_MODEL`
                environment variable.
            verdict_scores: Overrides for the score each verdict is worth.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.criteria = Criteria(criteria, DEFAULT_CRITERION)
        self.verdict_scores = self._resolve_scores(verdict_scores)
        self._client_spec = client
        self._client: LlmClient | None = None

    @staticmethod
    def _resolve_scores(overrides: dict[str, float] | None) -> dict[str, float]:
        """The default scores with `overrides` applied over them.

        Args:
            overrides: Per-verdict score overrides.

        Returns:
            The merged mapping from verdict to score.

        Raises:
            ValueError: If an override names a verdict this judge cannot reach.
                The reply schema only ever produces the three known words, so a
                score written for a fourth would sit dead forever.
        """
        unknown = set(overrides or ()) - set(Verdict)
        if unknown:
            known = ", ".join(sorted(Verdict))
            raise ValueError(
                f"unknown verdicts in verdict_scores: {sorted(unknown)}; "
                f"this judge decides one of: {known}"
            )
        return {**DEFAULT_VERDICT_SCORES, **(overrides or {})}

    def compute(self, node: EvalNode) -> MetricResult | None:
        """Judge every leaf beneath this node in one call.

        Args:
            node: The node to judge, with every leaf beneath it.

        Returns:
            The node's score with the per-field verdicts on its `extra`, keyed
            by absolute path, or `None` when there was nothing to judge.

        Raises:
            ValueError: If the sample carries no grounding `source`.
        """
        source = node.context.source
        if source is None:
            raise ValueError(
                "JudgeFaithfulness requires a grounding `source`; "
                "pass source=... to evaluate()"
            )
        if node.is_leaf() and self.criteria.addresses_fields:
            raise ValueError(
                f"JudgeFaithfulness on the field {node.path!r} was given criteria "
                "keyed by path, but a single field has nothing beneath it for the "
                "keys to address; pass one criterion string instead"
            )
        # Leaves only: an object or an array is not itself a claim about the
        # source — every claim it makes is one of the values underneath it, and
        # asking about the container as well would grade the same facts twice.
        judged = {
            relative_path(node.path, leaf.path): leaf
            for leaf in node.leaves()
            if leaf.is_present
        }
        if not judged:
            return None
        reply = self.client.generate_with_schema(
            build_prompt(
                source,
                [
                    (path, judged[path].actual, self.criteria.for_path(path))
                    for path in self._in_document_order(judged, node)
                ],
            ),
            JudgeReply,
            system=SYSTEM_PROMPT,
        )
        return self._collect(reply, judged)

    @classmethod
    def _document_order(cls, value: Any, prefix: str = "") -> Iterator[str]:
        """Relative paths of every leaf in `value`, as the document lists them.

        The tree sorts an object's children by name, which is right for a
        report but wrong for a prompt — the judge should see an amount next to
        its currency, not next to whatever shares its first letter.

        Args:
            value: The raw document or subtree to walk.
            prefix: Path accumulated so far; internal to the recursion.

        Yields:
            Each leaf's relative path, in document order.
        """
        if isinstance(value, dict):
            for key, child in value.items():
                yield from cls._document_order(
                    child, f"{prefix}.{key}" if prefix else key
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from cls._document_order(child, f"{prefix}[{index}]")
        else:
            yield prefix

    @classmethod
    def _in_document_order(
        cls, judged: dict[str, EvalNode], node: EvalNode
    ) -> list[str]:
        """The judged fields' paths, ordered as the document lists them.

        Both sides spell a path the same way, so they line up by string. A
        field the document does not carry — expected-only, or a value whose
        shape disagrees with the tree — keeps its place at the end instead of
        being dropped.

        Args:
            judged: The fields to order, keyed by relative path.
            node: The node they were collected from.

        Returns:
            The same paths, in document order.
        """
        if len(judged) < 2:
            return list(judged)
        order = {
            path: index for index, path in enumerate(cls._document_order(node.actual))
        }
        return sorted(judged, key=lambda path: order.get(path, len(order)))

    @property
    def client(self) -> LlmClient:
        """The judge's model, resolved once, on first use."""
        if self._client is None:
            self._client = resolve_client(self._client_spec)
        return self._client

    def _collect(
        self, reply: JudgeReply, judged: dict[str, EvalNode]
    ) -> MetricResult | None:
        """Turn the judge's answer into one score for the node it ran on."""
        verdicts: list[FieldJudgeVerdict] = []
        ruled: set[str] = set()
        weighted, total_weight = 0.0, 0.0
        for judgement in reply.verdicts:
            leaf = judged.get(judgement.path)
            # Drop a path the judge invented, and a second answer about one it
            # already ruled on. A field it never answered about stays out of
            # the mean rather than counting as a zero.
            if leaf is None or leaf.path in ruled:
                continue
            ruled.add(leaf.path)
            verdicts.append(
                FieldJudgeVerdict(
                    path=leaf.path, verdict=judgement.verdict, reason=judgement.reason
                )
            )
            weighted += self.verdict_scores[judgement.verdict] * leaf.weight
            total_weight += leaf.weight
        if not verdicts:
            return None
        summary = JudgeVerdict(
            score=weighted / total_weight if total_weight else 0.0, verdicts=verdicts
        )
        return MetricResult(summary.score, {"verdict": summary.model_dump()})
