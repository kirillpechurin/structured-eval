"""`ChatModelClient` — a LangChain-style model, duck-typed against a fake."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.exceptions import LlmInvocationError, LlmResponseFormatError

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    """The response schema passed to `generate_with_schema`."""

    score: float
    reason: str


class FakeChatModel:
    """Shaped like a LangChain chat model, dependency-free."""

    def __init__(self, reply: Any, *, model_name: str = "fake/model") -> None:
        """Pin the one reply every call returns, and start the call log empty."""
        self.reply = reply
        self.model_name = model_name
        self.seen: list[Any] = []
        self.bound_schema: type[BaseModel] | None = None

    def invoke(self, messages: Any) -> Any:
        """The pinned reply; `messages` is recorded in `seen` on the way past."""
        self.seen.append(messages)
        return self.reply


class StructuredChatModel(FakeChatModel):
    """A chat model that also offers `with_structured_output`."""

    def with_structured_output(self, schema: type[BaseModel]) -> FakeChatModel:
        """Record the bound schema and return self, as LangChain's does."""
        self.bound_schema = schema
        return self


class Reply:
    """A message object: its text sits behind `.content`."""

    def __init__(self, content: Any) -> None:
        """Wrap `content` as the one attribute the client reads."""
        self.content = content


@pytest.mark.parametrize(
    ("content", "text"),
    [
        ("plain text", "plain text"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["a", "b"], "ab"),
    ],
    ids=["string", "content-blocks", "string-list"],
)
def test_reply_is_flattened_to_text(content, text):
    assert ChatModelClient(FakeChatModel(Reply(content))).generate("hi") == text


def test_unknown_reply_body_is_stringified():
    """A provider returning neither text nor blocks must not crash the call."""
    assert ChatModelClient(FakeChatModel(Reply(42))).generate("hi") == "42"


def test_system_becomes_its_own_turn():
    """Not prompt text — a real message turn, which is the point of this adapter."""
    model = FakeChatModel(Reply("ok"))

    ChatModelClient(model).generate("judge this", system="be strict")

    assert model.seen == [[("system", "be strict"), ("human", "judge this")]]


def test_the_model_name_is_inferred():
    assert ChatModelClient(FakeChatModel(Reply("ok"))).model_name == "fake/model"


def test_the_model_name_falls_back_to_the_class():
    """A model exposing no usable identifier still gets a readable name."""

    class NamelessChatModel(FakeChatModel):
        model = 42  # present but not a string — must be skipped

    model = NamelessChatModel(Reply("ok"), model_name="")

    assert ChatModelClient(model).model_name == "NamelessChatModel"


def test_the_model_name_can_be_overridden():
    model = FakeChatModel(Reply("ok"))

    assert ChatModelClient(model, model_name="explicit").model_name == "explicit"


def test_a_model_without_invoke_is_rejected():
    with pytest.raises(TypeError, match=r"\.invoke\(\)"):
        ChatModelClient(object())


def test_a_failing_model_is_wrapped():
    class FailingChatModel(FakeChatModel):
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("429")

    with pytest.raises(LlmInvocationError, match="429"):
        ChatModelClient(FailingChatModel(None)).generate("judge")


def test_structured_output_is_preferred_when_available():
    model = StructuredChatModel(Verdict(score=0.25, reason="native"))

    result = ChatModelClient(model).generate_with_schema("judge", Verdict)

    assert model.bound_schema is Verdict  # went through with_structured_output
    assert result.reason == "native"


@pytest.mark.parametrize(
    "reply",
    [
        {"score": 0.25, "reason": "as dict"},
        '{"score": 0.25, "reason": "as text"}',
    ],
    ids=["dict", "json-string"],
)
def test_structured_output_accepts_either_reply_shape(reply):
    model = StructuredChatModel(reply)

    assert ChatModelClient(model).generate_with_schema("judge", Verdict).score == 0.25


def test_structured_output_mismatch_raises_response_format_error():
    model = StructuredChatModel({"score": "high"})

    with pytest.raises(LlmResponseFormatError):
        ChatModelClient(model).generate_with_schema("judge", Verdict)


def test_a_model_without_structured_output_falls_back_to_prompting():
    model = FakeChatModel(Reply('{"score": 0.5, "reason": "parsed"}'))

    result = ChatModelClient(model).generate_with_schema("judge", Verdict)

    assert result.reason == "parsed"


def test_the_fallback_appends_the_schema_instead_of_binding_it():
    model = FakeChatModel(Reply('{"score": 0.5, "reason": "parsed"}'))

    ChatModelClient(model).generate_with_schema("judge", Verdict)

    assert "JSON Schema" in model.seen[0][0][1]
