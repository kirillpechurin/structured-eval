"""The litellm-backed client against the real library.

`tests/unit/llm/test_litellm.py` drives a stub, so by construction it cannot
notice litellm changing shape underneath us. These tests pin the assumptions
that stub encodes — the response shape `completion` returns and the existence
of `supports_response_schema` — and are skipped when the extra is absent.
litellm's own `mock_response` keeps them offline: no API key, no network.
"""

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.integration


class Verdict(BaseModel):
    """The response schema the real model is asked to fill."""

    score: float
    reason: str


def test_generate_reads_the_real_response_shape() -> None:
    pytest.importorskip("litellm")
    from structured_eval.llm.litellm import LiteLlmClient

    client = LiteLlmClient("openai/gpt-4o-mini", mock_response="a plain answer")

    assert client.generate("judge this") == "a plain answer"


def test_schema_path_parses_a_real_response() -> None:
    pytest.importorskip("litellm")
    from structured_eval.llm.litellm import LiteLlmClient

    client = LiteLlmClient(
        "openai/gpt-4o-mini", mock_response='{"score": 1.0, "reason": "grounded"}'
    )

    assert client.generate_with_schema("judge this", Verdict).score == 1.0


def test_supports_response_schema_is_still_the_routing_helper() -> None:
    """The native/fallback switch calls this by name — it must exist and answer."""
    litellm = pytest.importorskip("litellm")

    assert isinstance(litellm.supports_response_schema(model="openai/gpt-4o"), bool)
