"""The provider-neutral seam every LLM-backed feature calls.

``LlmClient`` is deliberately tiny and knows no provider: one abstract method,
``generate(prompt) -> str``. Everything above it (judge metrics, LLM array
alignment) talks to this and nothing else, so the core package stays free of
provider SDKs.

The second method, ``generate_with_schema``, is the flexible half. It has a
**working default** built on ``generate``: the JSON Schema is appended to the
prompt, and the reply is extracted and validated on our side. An implementation
whose provider supports native structured outputs overrides it and gets a format
*guarantee* instead of a request. Either way callers see the same contract — a
validated model instance — so a ten-line wrapper is a complete client.
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
    """What a judge metric needs from a model. Implement ``generate``; done.

    ``model_name`` is free-form and only descriptive — it records *which* model
    produced a judgement, so set it to something a reader can act on.

    ``schema_prompt`` / ``parse_reply`` / ``validate_reply`` are the three steps
    of the default schema strategy, split so a subclass can replace one of them
    without reimplementing ``generate_with_schema``: reword the instruction,
    change how a reply is sliced, or hand over an already-parsed payload.
    """

    model_name: str = ""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's reply as plain text."""

    def generate_with_schema[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        """Return a reply validated against ``schema``.

        The default asks for JSON in the prompt and parses the answer. Override
        when the provider can constrain generation to the schema itself.
        """
        raw = self.generate(self.schema_prompt(prompt, schema), system=system)
        return self.parse_reply(raw, schema)

    def schema_prompt(self, prompt: str, schema: type[BaseModel]) -> str:
        """Append ``schema`` to ``prompt`` as a JSON-only instruction."""
        rendered = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        return (
            f"{prompt}\n\n"
            "Reply with a single JSON object and nothing else — no prose, no code "
            f"fences. It must validate against this JSON Schema:\n{rendered}"
        )

    def parse_reply[T: BaseModel](self, raw: str, schema: type[T]) -> T:
        """Read a text reply as ``schema``, tolerating fences and prose.

        Providers that constrain generation but still return a JSON *string*
        end here too — only the guarantee differs, not the parsing.
        """
        try:
            return schema.model_validate_json(self._extract_json(raw))
        except ValidationError as exc:
            raise LlmResponseFormatError(
                f"model reply does not match {schema.__name__}: {self._preview(raw)}"
            ) from exc

    def validate_reply[T: BaseModel](self, payload: object, schema: type[T]) -> T:
        """Validate an already-parsed reply (a mapping or a model) against ``schema``."""
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
        text = raw.strip()
        return repr(
            text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"
        )
