"""Adapter for a LangChain-style chat model — duck-typed, no dependency.

Nothing here is imported from LangChain: the adapter only calls ``.invoke()``
and, when present, ``.with_structured_output()``. So a model the user has
already configured — proxy, retries, callbacks, tracing and all — becomes a
client without adding a package, and every provider LangChain supports comes
along with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmInvocationError

if TYPE_CHECKING:
    from pydantic import BaseModel


class ChatModelClient(LlmClient):
    """Wraps a chat model — anything with ``.invoke()``.

    Over wrapping the same model in a ``CallableClient`` this adds the things a
    naive ``lambda p: model.invoke(p).content`` gets wrong: ``system`` becomes a
    real message turn instead of prompt text, replies arriving as content blocks
    are flattened, and schema requests go through ``.with_structured_output()``
    so the provider's native structured outputs are used instead of the
    inherited prompt-and-parse fallback.
    """

    # Attributes a chat model may carry its identifier under, most specific first.
    _NAME_ATTRS = ("model_name", "model", "model_id")

    def __init__(self, chat_model: Any, *, model_name: str | None = None) -> None:
        if not callable(getattr(chat_model, "invoke", None)):
            raise TypeError(
                f"chat model must expose an .invoke() method, got {type(chat_model)!r}"
            )
        self._model = chat_model
        self.model_name = model_name or self._infer_model_name(chat_model)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        reply = self._invoke(self._model, prompt, system)
        return self._as_text(getattr(reply, "content", reply))

    def generate_with_schema[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        bind = getattr(self._model, "with_structured_output", None)
        if bind is None:
            return super().generate_with_schema(prompt, schema, system=system)
        reply = self._invoke(bind(schema), prompt, system)
        if isinstance(reply, str):
            return self.parse_reply(reply, schema)
        return self.validate_reply(reply, schema)

    def _invoke(self, model: Any, prompt: str, system: str | None) -> Any:
        messages: list[tuple[str, str]] = [("human", prompt)]
        if system:
            messages.insert(0, ("system", system))
        try:
            return model.invoke(messages)
        except Exception as exc:
            raise LlmInvocationError(
                f"chat model {self.model_name!r} failed: {exc}"
            ) from exc

    @classmethod
    def _infer_model_name(cls, chat_model: Any) -> str:
        for attr in cls._NAME_ATTRS:
            value = getattr(chat_model, attr, None)
            if isinstance(value, str) and value:
                return value
        return type(chat_model).__name__

    @staticmethod
    def _as_text(content: Any) -> str:
        """Flatten a reply body to text — providers may return content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
                if isinstance(block, str | dict)
            ]
            return "".join(parts)
        return str(content)
