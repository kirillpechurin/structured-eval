"""What the judge is asked, and the shape it must answer in.

Contains the wording of the question and the vocabulary of the answer.
"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.judge_faithfulness.schemas import Verdict
from structured_eval.metrics.utils.value import render_value

# Used for any field the caller wrote no criterion for. Deliberately bland: a
# field-specific criterion is the judge's most valuable input, and this is the
# honest default when nobody supplied one.
DEFAULT_CRITERION = "the value must be stated by, or follow directly from, the source"

SYSTEM_PROMPT = (
    "You verify structured data extracted from a source text. For each field you "
    "decide whether the source backs the extracted value. The source is the only "
    "truth available to you: never rely on outside knowledge, plausibility, or "
    "what the value ought to be."
)

_VERDICT_RULES = f"""\
Give each field exactly one verdict:
- "{Verdict.SUPPORTED}": the source states the value, or the value follows directly from it.
- "{Verdict.CONTRADICTED}": the source states something incompatible with the value.
- "{Verdict.NOT_STATED}": the source neither states nor contradicts the value — nothing \
in it backs the value up.

A null value is a claim too: it asserts that the source gives no value for that \
field. Judge that claim, not the absence — "{Verdict.SUPPORTED}" when the source indeed \
offers none, "{Verdict.CONTRADICTED}" when the source plainly states one.

Answer about every field listed, reusing its `path` exactly as given. Fill in \
`reason` only when the verdict is not "{Verdict.SUPPORTED}"; leave it empty otherwise, \
and keep it to one sentence naming the part of the source you relied on."""


def build_prompt(source: str, fields: list[tuple[str, Any, str]]) -> str:
    """Assemble the single request covering every judged field.

    The fields travel in one prompt on purpose: the judge sees them together,
    so a value that only makes sense next to its neighbours is not ruled on in
    isolation, and a wide object costs one call rather than one per field.

    Args:
        source: The grounding text the judge rules against.
        fields: One `(path, value, criterion)` per judged field.

    Returns:
        The assembled prompt.
    """
    listed = "\n".join(
        f"- path: {path}\n  value: {render_value(value)}\n  criterion: {criterion}"
        for path, value, criterion in fields
    )
    return (
        f'Source text:\n"""\n{source}\n"""\n\n'
        f"Fields extracted from that source:\n{listed}\n\n"
        f"{_VERDICT_RULES}"
    )
