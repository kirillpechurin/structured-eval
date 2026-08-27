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

    Identical in shape to `FieldJudgeVerdict`, and the duplication is
    deliberate — the two answer to opposite constraints:

    - This is the **wire** model: `verdict` is a closed enum, which is what
      keeps the reply schema down to three words the model chooses between.
    - `FieldJudgeVerdict` is the **report** model shared by every judge, so its
      `verdict` stays an open string.

    Attributes:
        path: The field the judge was asked about.
        verdict: What it decided.
        reason: Why, when the verdict is not `supported`.
    """

    path: str
    verdict: Verdict
    reason: str = ""


class JudgeReply(BaseModel):
    """The judge's whole answer — one entry per field, in a single call.

    Attributes:
        verdicts: One ruling per field the judge was asked about.
    """

    verdicts: list[JudgedField] = Field(default_factory=list)
