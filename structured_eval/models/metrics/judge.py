"""The verdict models an LLM-judge metric reports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldJudgeVerdict(BaseModel):
    """One judged field: which node, what the judge decided, and why.

    Every claim is already a node, so a verdict points at the exact field it is
    about and the report can key on it.

    Attributes:
        path: The judged node's absolute path.
        verdict: What the judge decided for that field.
        reason: Why, when the judge gave a reason.
    """

    path: str
    verdict: str
    reason: str = ""


class JudgeVerdict(BaseModel):
    """A judge's full result for one node: its own score plus per-field detail.

    Attributes:
        score: The node-level number the metric reports for the judged node.
        reason: Why, when the judge explains its node-level verdict — not a
            concatenation of the per-field reasons.
        verdicts: One entry per field the judge ruled on.
    """

    score: float
    reason: str = ""
    verdicts: list[FieldJudgeVerdict] = Field(default_factory=list)
