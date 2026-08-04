"""The reply schema the faithfulness judge is asked to fill in."""

import enum

from pydantic import BaseModel, Field


class Verdict(enum.StrEnum):
    """How the judge ruled on one field against the source.

    Attributes:
        SUPPORTED: The source backs the value.
        NOT_STATED: The source neither backs nor contradicts it.
        CONTRADICTED: The source says otherwise.
    """

    SUPPORTED = "supported"
    NOT_STATED = "not_stated"
    CONTRADICTED = "contradicted"


class JudgedField(BaseModel):
    """One field as the judge rules on it, keyed by the path it was asked about.

    Three fields identical to ``FieldJudgeVerdict``, and the duplication is
    deliberate — they answer to opposite constraints. This one is the **wire**
    model: ``verdict`` has to be a closed enum, because that is what keeps the
    reply schema down to three words the model must choose between. The other
    is the **report** model, shared by every judge, and its ``verdict`` has to
    stay an open string — closing it would tie the family contract to
    faithfulness' own vocabulary. Merging them means picking one of the two,
    and both are load-bearing.
    """

    path: str
    verdict: Verdict
    reason: str = ""


class JudgeReply(BaseModel):
    """The judge's whole answer — one entry per field, in a single call."""

    verdicts: list[JudgedField] = Field(default_factory=list)
