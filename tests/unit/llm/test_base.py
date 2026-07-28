"""The LlmClient contract: one abstract method, schema support for free."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmResponseFormatError

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    score: float
    reason: str


class ScriptedClient(LlmClient):
    """A complete client in the smallest form the interface allows."""

    model_name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        '{"score": 0.5, "reason": "partly grounded"}',
        '```json\n{"score": 0.5, "reason": "partly grounded"}\n```',
        '```\n{"score": 0.5, "reason": "partly grounded"}\n```',
        'Here is my verdict:\n{"score": 0.5, "reason": "partly grounded"}\nDone.',
        '  \n {"score": 0.5, "reason": "partly grounded"}  ',
    ],
    ids=["bare", "json-fence", "plain-fence", "narrated", "whitespace"],
)
def test_default_schema_path_tolerates_reply_wrapping(reply):
    result = ScriptedClient(reply).generate_with_schema("judge this", Verdict)

    assert (result.score, result.reason) == (0.5, "partly grounded")


def test_default_schema_path_extends_the_original_prompt():
    client = ScriptedClient('{"score": 1.0, "reason": "ok"}')

    client.generate_with_schema("judge this", Verdict)

    prompt = client.prompts[0]
    assert prompt.startswith("judge this")
    # The schema itself is in the prompt — that is what makes the fallback work.
    assert json.dumps(Verdict.model_json_schema(), ensure_ascii=False) in prompt


def test_default_schema_path_passes_system_through_untouched():
    client = ScriptedClient('{"score": 1.0, "reason": "ok"}')

    client.generate_with_schema("judge this", Verdict, system="be strict")

    assert client.systems == ["be strict"]


def test_a_client_without_generate_cannot_be_built():
    """``generate`` is the whole contract — nothing else may be left unimplemented."""

    class Incomplete(LlmClient):
        pass

    with pytest.raises(TypeError, match="generate"):
        Incomplete()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "reply",
    ["no json here at all", "{ unbalanced", '{"score": "high", "reason": "x"}'],
    ids=["no-object", "truncated", "wrong-types"],
)
def test_unusable_reply_raises_response_format_error(reply):
    with pytest.raises(LlmResponseFormatError):
        ScriptedClient(reply).generate_with_schema("judge this", Verdict)


def test_response_format_error_quotes_the_reply():
    with pytest.raises(LlmResponseFormatError, match="sorry, I cannot"):
        ScriptedClient("").parse_reply("sorry, I cannot", Verdict)


def test_long_reply_is_truncated_in_the_error():
    with pytest.raises(LlmResponseFormatError, match="…"):
        ScriptedClient("").parse_reply("x" * 500, Verdict)


def test_schema_prompt_demands_bare_json():
    prompt = ScriptedClient("").schema_prompt("rate it", Verdict)

    assert "no prose, no code fences" in prompt


def test_validate_reply_accepts_a_parsed_mapping():
    result = ScriptedClient("").validate_reply({"score": 1.0, "reason": "ok"}, Verdict)

    assert result.score == 1.0


def test_validate_reply_rejects_a_mismatch():
    with pytest.raises(LlmResponseFormatError):
        ScriptedClient("").validate_reply({"score": "high"}, Verdict)
