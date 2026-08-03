from __future__ import annotations

from pydantic import BaseModel, Field


class FieldJudgeVerdict(BaseModel):
    """One judged field: which node, what the judge decided, and why.

    ``path`` is the node's absolute path — the identity that makes a structured
    judge worth more than a prose one. Free-text judges have to quote the claim
    they graded; here every claim is already a node, so a verdict points at the
    exact field it is about and the report can key on it.
    """

    path: str
    verdict: str
    reason: str = ""


class JudgeVerdict(BaseModel):
    """A judge's full result for one node: its own score plus per-field detail.

    ``score`` is the node-level number the metric reports for the judged node
    itself; ``verdicts`` carries one entry per field it ruled on. ``reason``
    explains the node-level verdict when a judge produces one — it is not a
    concatenation of the per-field reasons.
    """

    score: float
    reason: str = ""
    verdicts: list[FieldJudgeVerdict] = Field(default_factory=list)
