"""Adapter for a LangChain-style chat model — duck-typed, no dependency.

Nothing here is imported from LangChain: the adapter only calls `.invoke()`
and, when present, `.with_structured_output()`. A model the user has already
configured — proxy, retries, callbacks, tracing and all — becomes a client
without adding a package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmInvocationError

if TYPE_CHECKING:
    from pydantic import BaseModel


class ChatModelClient(LlmClient):
    """Wraps a chat model — anything with `.invoke()`.

    Over wrapping the same model in a `CallableClient` this adds the three
    things a naive `lambda p: model.invoke(p).content` gets wrong:

    - `system` becomes a real message turn instead of prompt text;
    - a reply arriving as content blocks is flattened to text;
    - a schema request goes through `.with_structured_output()`, so the
      provider's native structured outputs are used instead of the
      prompt-and-parse fallback inherited from `LlmClient`.

    Example:
        >>> from langchain_openai import ChatOpenAI  # doctest: +SKIP
        >>> from structured_eval.llm import ChatModelClient
        >>> model = ChatOpenAI(model="gpt-4o", temperature=0)  # doctest: +SKIP
        >>> client = ChatModelClient(model)  # doctest: +SKIP
        >>> client.model_name  # off the model itself  # doctest: +SKIP
        'gpt-4o'
        >>> client.generate("Is the summary faithful?",
        ...                 system="Answer in one line.")  # doctest: +SKIP
        'Yes — the total and the date both appear in the source.'
    """

    # Attributes a chat model may carry its identifier under, most specific first.
    _NAME_ATTRS = ("model_name", "model", "model_id")

    def __init__(self, chat_model: Any, *, model_name: str | None = None) -> None:
        """Adopt a chat model, taking its name from the model when not given.

        Args:
            chat_model: Any object exposing `.invoke()`.
            model_name: What to record as the model. Defaults to the first of
                `model_name` / `model` / `model_id` the object carries, and to
                its class name when it carries none.

        Raises:
            TypeError: If `chat_model` has no callable `.invoke()`.
        """
        if not callable(getattr(chat_model, "invoke", None)):
            raise TypeError(
                f"chat model must expose an .invoke() method, got {type(chat_model)!r}"
            )
        self._model = chat_model
        self.model_name = model_name or self._infer_model_name(chat_model)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Invoke the model with real message turns and flatten the reply.

        Args:
            prompt: The user turn.
            system: The system turn, sent as its own message when set.

        Returns:
            The reply as text, with content blocks joined.

        Raises:
            LlmInvocationError: If the model raised; the original exception is
                kept as `__cause__`.
        """
        reply = self._invoke(self._model, prompt, system)
        return self._as_text(getattr(reply, "content", reply))

    def generate_with_schema[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        """Ask through `.with_structured_output()` when the model offers it.

        A model without that method falls back to the inherited prompt-and-parse
        path, so nothing is lost by wrapping a plainer one.

        Args:
            prompt: The user turn.
            schema: The model the reply must validate against.
            system: The system turn, sent as its own message when set.

        Returns:
            An instance of `schema`, whether the provider returned a parsed
            object or a JSON string.

        Raises:
            LlmInvocationError: If the model raised.
            LlmResponseFormatError: If the reply does not match `schema`.
        """
        bind = getattr(self._model, "with_structured_output", None)
        if bind is None:
            return super().generate_with_schema(prompt, schema, system=system)
        reply = self._invoke(bind(schema), prompt, system)
        if isinstance(reply, str):
            return self.parse_reply(reply, schema)
        return self.validate_reply(reply, schema)

    def _invoke(self, model: Any, prompt: str, system: str | None) -> Any:
        """One chat-model call, with any provider failure turned into ours."""
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
        """The model's own name, however this provider spells the attribute."""
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
