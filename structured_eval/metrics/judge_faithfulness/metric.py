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

# What each verdict is worth. ``not_stated`` scoring 0.0 is where we part with
# deepeval, whose ``idk`` counts as faithful: in RAG "the context does not
# mention it" is benign, while for a *structured extraction* a value nobody
# stated is exactly the fabrication the metric exists to catch.
DEFAULT_VERDICT_SCORES: dict[str, float] = {
    Verdict.SUPPORTED: 1.0,
    Verdict.NOT_STATED: 0.0,
    Verdict.CONTRADICTED: 0.0,
}


class JudgeFaithfulness(AnyNodeMetric):
    """Is each field beneath this node grounded in the sample's ``source``?

    Attach it to any node — the document root, one nested object, an array, or a
    single field — and it takes every leaf beneath that node, asks one LLM call
    whether the source backs each value, and rules on each:
    ``supported`` / ``not_stated`` / ``contradicted``, with a reason for the ones
    that are not supported. Each verdict is scored through ``verdict_scores``,
    and the node scores their mean, weighted by field ``weight``.

    The score lands on the **judged node**, and the judge's full result rides
    along in its ``extra`` as one ``JudgeVerdict`` — the node's own score, and
    a verdict per field keyed by the absolute path it is about::

        report.field_scores["line_items"].metrics["judge_faithfulness"]
            .extra["verdict"]["verdicts"]
        # [{"path": "line_items[2].sku", "verdict": "not_stated", "reason": "no X-3"}]

        report.metrics["judge_faithfulness"].extra_values("verdict")
        # one JudgeVerdict per judged node, wherever the judges were hung

    A verdict does **not** become the score of the field it names: to grade
    ``line_items[2].sku`` itself, hang a judge on that field. Where to hang the
    judge is therefore a real choice — one call for a whole subtree, with the
    detail as evidence, or one call per field, with the detail as a score.

    Why one call for the whole subtree: the fields are judged together, so a
    value that only makes sense beside its neighbours is not ruled on in
    isolation — and a wide object costs one call, not one per field. Attached to
    a lone field it costs one call for that field, which is the point of hanging
    it there: pay the model only where nothing cheaper can decide.

    ``criteria`` says what "faithful" means. Either one rule for everything::

        JudgeFaithfulness("the value must follow from the stated age")

    or a rule per field, keyed by paths **relative to the judged node** — so the
    same configuration works wherever it is hung, with ``[*]`` for array
    elements::

        JudgeFaithfulness({"vendor": "...", "line_items[*].sku": "..."})

    Fields with no entry fall back to a generic formulation. A mapping on a
    single field is refused: there is nothing beneath a leaf for the keys to
    address.

    Only what the output actually produced is judged — the same rule the
    reference faithfulness metrics follow, where the claims come from the
    answer and the ground truth never enters. An explicit ``null`` **is** such
    a claim ("the source gives no value here"), true (``supported``) or false
    (``contradicted``) like any other; a field simply absent from the output
    claims nothing and is left out. Missing it is a recall failure, which
    ``Presence`` / ``CoverageLeafScore`` already measure — keeping the two
    apart is what stops one number from meaning both.

    Costs money and time and is not deterministic — it runs only where it is
    explicitly configured, never by default. Needs a grounding ``source``;
    without one faithfulness is undefined and this raises rather than quietly
    scoring nothing. The client comes from ``client=`` or, unset, from
    ``STRUCTURED_EVAL_LLM_MODEL`` (see ``structured_eval.llm``); it is resolved
    on first use, so building a config costs no credentials.
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
        super().__init__(name=name)
        self.criteria = Criteria(criteria, DEFAULT_CRITERION)
        self.verdict_scores = self._resolve_scores(verdict_scores)
        self._client_spec = client
        self._client: LlmClient | None = None

    @staticmethod
    def _resolve_scores(overrides: dict[str, float] | None) -> dict[str, float]:
        """The default scores with ``overrides`` applied over them.

        An override naming a verdict this judge cannot reach is a configuration
        error rather than a no-op: the reply schema only ever produces the three
        words above, so a score written for a fourth would sit dead forever.
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
        # Present leaves only: a node the output never produced (it exists
        # because `expected` has the key) claims nothing, so there is nothing
        # to check it against. That gap is recall, which `Presence` /
        # `CoverageLeafScore` already measure.
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
        """Relative paths of every leaf in ``value``, as the document lists them.

        The tree sorts an object's children by name, which is right for a report
        but wrong for a prompt: the judge is asked to read the fields together,
        and "together" means as the document wrote them — an amount next to its
        currency, not next to whatever happens to share its first letter.
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

        Both sides spell a path the same way — the tree joins a child's key onto
        its parent's path, ``_document_order`` joins the same keys walking the
        raw value — so the two line up by string. A field the document does not
        carry (expected-only, or a value whose shape disagrees with the tree)
        keeps its place at the end instead of being dropped; ``sorted`` is
        stable, so those stay in the order the tree gave them.

        A single field (a judged leaf, whose own relative path is its name and
        never appears in the walk) needs no ordering at all.
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
        """Turn the judge's answer into one score for the node it ran on.

        The score is the mean of the verdicts, weighted by each field's
        ``weight``, so a field the caller marked important pulls harder here
        too. The verdicts themselves ride along in ``extra`` — that detail is
        the whole reason to pay a model rather than compare strings.

        They travel as one whole ``JudgeVerdict`` rather than a bare list, so
        the node's own verdict and the fields backing it stay one object;
        ``MetricCollection.extra_values("verdict")`` then gathers one such
        object per judged node across the tree.

        A field the judge said nothing about is absent from the mean rather
        than counted as a zero — silence is missing evidence, not a failed
        check — and a path it invented is dropped. Both keep a sloppy reply
        from quietly inventing numbers. Answering about nothing at all yields
        ``None``: the metric had nothing to say, exactly like any other metric
        that opts out.

        Verdicts are asked for by path relative to the judged node and reported
        by the absolute one: the prompt speaks the language of the subtree, the
        report speaks the language of the document.
        """
        verdicts: list[FieldJudgeVerdict] = []
        ruled: set[str] = set()
        weighted, total_weight = 0.0, 0.0
        for judgement in reply.verdicts:
            leaf = judged.get(judgement.path)
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
