"""The provider-neutral seam every LLM-backed feature calls.

`LlmClient` knows no provider: judge metrics and LLM array alignment talk to it
and nothing else, which is what keeps the core package free of provider SDKs.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

from structured_eval.llm.exceptions import LlmResponseFormatError

# How much of an unusable reply to quote back in an error message.
_PREVIEW_CHARS = 200

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LlmClient(ABC):
    """What a judge metric needs from a model. Implement `generate`; done.

    `model_name` is free-form and only descriptive — it records *which* model
    produced a judgement, so set it to something a reader can act on.

    `schema_prompt` / `parse_reply` / `validate_reply` are the three steps of
    the default schema strategy, split so a subclass can replace one of them
    without reimplementing `generate_with_schema`: reword the instruction,
    change how a reply is sliced, or hand over an already-parsed payload.

    Attributes:
        model_name: Which model produced a reply, for the report to record.

    Example:
        >>> from structured_eval.llm import LlmClient
        >>> class GatewayClient(LlmClient):
        ...     model_name = "internal/reviewer-v3"
        ...     def generate(self, prompt: str, *, system: str | None = None) -> str:
        ...         return internal_gateway.chat(system, prompt)
        >>> client = GatewayClient()
        >>> client.generate("Is the summary faithful?")  # doctest: +SKIP
        'Yes — the total and the date both appear in the source.'

    Implementing that one method is enough for both: `generate_with_schema`
    appends the JSON Schema to the prompt and validates the reply on our side.
    """

    model_name: str = ""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's reply as plain text.

        Args:
            prompt: The user turn.
            system: The system turn, when the caller sets one.

        Returns:
            The reply text, as the provider gave it.
        """

    def generate_with_schema[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        """Return a reply validated against `schema`.

        The default asks for JSON in the prompt and parses the answer. Override
        when the provider can constrain generation to the schema itself.

        Args:
            prompt: The user turn.
            schema: The model the reply must validate against.
            system: The system turn, when the caller sets one.

        Returns:
            An instance of `schema`.

        Raises:
            LlmResponseFormatError: If the reply cannot be read as `schema`.
        """
        raw = self.generate(self.schema_prompt(prompt, schema), system=system)
        return self.parse_reply(raw, schema)

    def schema_prompt(self, prompt: str, schema: type[BaseModel]) -> str:
        """Append `schema` to `prompt` as a JSON-only instruction.

        Args:
            prompt: The instruction to extend.
            schema: The model whose JSON Schema is spelled out for the model.

        Returns:
            The prompt, followed by the schema and the demand to answer with
            nothing but a JSON object.
        """
        rendered = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        return (
            f"{prompt}\n\n"
            "Reply with a single JSON object and nothing else — no prose, no code "
            f"fences. It must validate against this JSON Schema:\n{rendered}"
        )

    def parse_reply[T: BaseModel](self, raw: str, schema: type[T]) -> T:
        """Read a text reply as `schema`, tolerating fences and prose.

        Providers that constrain generation but still return a JSON *string*
        end here too — only the guarantee differs, not the parsing.

        Args:
            raw: The reply text.
            schema: The model to validate the extracted JSON against.

        Returns:
            An instance of `schema`.

        Raises:
            LlmResponseFormatError: If the reply holds no JSON object, or the
                object does not validate. The message quotes the reply.
        """
        try:
            return schema.model_validate_json(self._extract_json(raw))
        except ValidationError as exc:
            raise LlmResponseFormatError(
                f"model reply does not match {schema.__name__}: {self._preview(raw)}"
            ) from exc

    def validate_reply[T: BaseModel](self, payload: object, schema: type[T]) -> T:
        """Validate an already-parsed reply — a mapping or a model — as `schema`.

        Args:
            payload: What the provider returned, already parsed.
            schema: The model to validate it against.

        Returns:
            An instance of `schema`.

        Raises:
            LlmResponseFormatError: If `payload` does not match `schema`.
        """
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise LlmResponseFormatError(
                f"model reply does not match {schema.__name__}: "
                f"{self._preview(str(payload))}"
            ) from exc

    @classmethod
    def _extract_json(cls, raw: str) -> str:
        """Slice the JSON object out of a reply that may be fenced or narrated."""
        text = raw.strip()
        if fenced := _JSON_FENCE.search(text):
            text = fenced.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LlmResponseFormatError(
                f"no JSON object in model reply: {cls._preview(raw)}"
            )
        return text[start : end + 1]

    @staticmethod
    def _preview(raw: str) -> str:
        """The head of a reply, quoted, for an error message to carry."""
        text = raw.strip()
        return repr(
            text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"
        )
