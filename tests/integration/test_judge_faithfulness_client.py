"""JudgeFaithfulness driven by a real `LlmClient`, not a stand-in.

`tests/unit/metrics/test_judge_faithfulness.py` fakes the client and
`test_litellm_client.py` exercises litellm without the judge — so between them
nothing checks the seam the two meet at: the schema handed to the provider and
the reply parsed back into verdicts.

litellm's `mock_response` keeps these offline: no API key, no network.
"""

import json
from typing import Any

import pytest

from structured_eval import evaluate
from structured_eval.metrics import JudgeFaithfulness
from structured_eval.models import EvalConfig, EvalReport, ObjectFieldConfig

pytestmark = pytest.mark.integration

MODEL = "openai/gpt-4o-mini"
SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, issued in Berlin."
ACTUAL = {"vendor": "Acme Corp", "total": 999.0}

REPLY = json.dumps(
    {
        "verdicts": [
            {"path": "vendor", "verdict": "supported", "reason": "named in the text"},
            {"path": "total", "verdict": "contradicted", "reason": "the text says 100"},
        ]
    }
)


def _judged_by(reply: str) -> EvalReport:
    """Evaluate ACTUAL with a judge backed by a real litellm client."""
    from structured_eval.llm.litellm import LiteLlmClient

    metric = JudgeFaithfulness(client=LiteLlmClient(MODEL, mock_response=reply))
    report = evaluate(
        ACTUAL,
        None,
        EvalConfig(root=ObjectFieldConfig(metrics=[metric])),
        source=SOURCE,
    )
    assert isinstance(report, EvalReport)
    return report


def test_a_real_client_returns_verdicts_the_judge_can_score() -> None:
    pytest.importorskip("litellm")

    report = _judged_by(REPLY)

    result = report.field_scores["$"].metrics["judge_faithfulness"]
    assert float(result) == pytest.approx(0.5)  # supported 1.0 + contradicted 0.0
    verdicts = {v["path"]: v["verdict"] for v in result.extra["verdict"]["verdicts"]}
    assert verdicts == {"vendor": "supported", "total": "contradicted"}


def test_a_reply_that_is_not_the_asked_for_shape_is_reported_as_such() -> None:
    # The judge asks for a schema, but a provider may answer with prose anyway.
    # That has to surface as a response-format failure rather than as a score of
    # zero, which would read as "the document is unfaithful".
    pytest.importorskip("litellm")
    from structured_eval.llm.exceptions import LlmResponseFormatError

    with pytest.raises(LlmResponseFormatError):
        _judged_by("I think it all looks fine, honestly.")


def test_a_model_string_routes_to_the_litellm_client() -> None:
    # The keyless path: `client="provider/model"` is what a user writes, and it
    # has to land on the real class rather than on anything test-shaped.
    pytest.importorskip("litellm")
    from structured_eval.llm.litellm import LiteLlmClient

    client: Any = JudgeFaithfulness(client=MODEL).client

    assert isinstance(client, LiteLlmClient)
    assert client.model_name == MODEL
