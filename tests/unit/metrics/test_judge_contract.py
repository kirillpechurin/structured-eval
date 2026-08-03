"""The shape an LLM judge reports in: ``FieldJudgeVerdict`` / ``JudgeVerdict``.

A judge produces one score for the node it ran on plus a verdict per field
underneath it. The verdicts travel to the report inside ``MetricResult.extra``,
which is serialized with the report — so what is pinned here is the round trip
and the defaults callers rely on, independent of *what* the judge judges.
"""

import pytest

from structured_eval.models import FieldJudgeVerdict, JudgeVerdict

pytestmark = pytest.mark.unit


def test_field_verdict_defaults_reason_to_empty() -> None:
    v = FieldJudgeVerdict(path="invoice.total", verdict="supported")
    assert (v.path, v.verdict, v.reason) == ("invoice.total", "supported", "")


def test_verdict_vocabulary_is_not_constrained() -> None:
    # The family contract fixes the shape, never the words: a judge that is not
    # about faithfulness brings its own categories without touching the model.
    assert FieldJudgeVerdict(path="$", verdict="complete").verdict == "complete"


def test_judge_verdict_defaults_to_no_verdicts() -> None:
    jv = JudgeVerdict(score=1.0)
    assert jv.verdicts == []
    assert jv.reason == ""


def test_judge_verdict_round_trips() -> None:
    jv = JudgeVerdict(
        score=0.5,
        reason="one field is not in the source",
        verdicts=[
            FieldJudgeVerdict(path="a", verdict="supported"),
            FieldJudgeVerdict(path="b", verdict="not_stated", reason="absent"),
        ],
    )
    assert JudgeVerdict.model_validate(jv.model_dump()) == jv


def test_a_verdict_survives_the_dump_into_metric_extra() -> None:
    # How it actually reaches a reader: dumped into `extra`, read back by key.
    jv = JudgeVerdict(
        score=0.0,
        verdicts=[
            FieldJudgeVerdict(path="a", verdict="contradicted", reason="says 40")
        ],
    )
    extra = {"judge_verdict": jv.model_dump()}
    assert extra["judge_verdict"]["verdicts"][0]["reason"] == "says 40"
