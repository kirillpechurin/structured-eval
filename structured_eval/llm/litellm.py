"""The batteries-included client: any provider LiteLLM speaks, one string.

Behind the ``litellm`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmInvocationError

if TYPE_CHECKING:
    from pydantic import BaseModel

_INSTALL_HINT = (
    "litellm is required for LiteLlmClient. "
    "Install it with: pip install 'structured-eval[litellm]'"
)


class LiteLlmClient(LlmClient):
    """Calls `litellm.completion` for a `"provider/model"` identifier.

    `temperature` / `max_tokens` / `timeout` are passed through only when set,
    so each provider's own defaults apply otherwise; any further keyword is
    forwarded to `litellm.completion` untouched — `api_base` for a local
    endpoint, `num_retries`, and so on.

    Example:
        >>> from structured_eval.llm import LiteLlmClient
        >>> client = LiteLlmClient("openai/gpt-4o", temperature=0.0, timeout=30)
        >>> client.model_name
        'openai/gpt-4o'
        >>> client.generate("Is the summary faithful?")  # doctest: +SKIP
        'Yes — the total and the date both appear in the source.'
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **params: Any,
    ) -> None:
        """Pin the model and the call parameters every request carries.

        Args:
            model: A `"provider/model"` identifier litellm recognises.
            temperature: Sampling temperature, when set.
            max_tokens: Cap on the reply length, when set.
            timeout: Per-call timeout in seconds, when set.
            **params: Any further `litellm.completion` keyword.

        Raises:
            ValueError: If `model` is empty.
        """
        if not model:
            raise ValueError("model must be a non-empty 'provider/model' string")
        self.model_name = model
        optional = (
            ("temperature", temperature),
            ("max_tokens", max_tokens),
            ("timeout", timeout),
        )
        self._params: dict[str, Any] = {k: v for k, v in optional if v is not None}
        self._params.update(params)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Complete `prompt` through litellm and return the reply text.

        Args:
            prompt: The user turn.
            system: The system turn, sent as its own message when set.

        Returns:
            The completion's text content.

        Raises:
            ImportError: If litellm is not installed.
            LlmInvocationError: If the call failed, or the response carries no
                usable content.
        """
        return self._content(self._call(self._messages(prompt, system)))

    def generate_with_schema[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        """Constrain the reply to `schema` when the model supports it.

        Models that cannot constrain generation fall back to the inherited
        prompt-and-parse path.

        Args:
            prompt: The user turn.
            schema: The model the reply must validate against.
            system: The system turn, sent as its own message when set.

        Returns:
            An instance of `schema`.

        Raises:
            ImportError: If litellm is not installed.
            LlmInvocationError: If the call failed.
            LlmResponseFormatError: If the reply does not match `schema`.
        """
        if not self._supports_schema():
            return super().generate_with_schema(prompt, schema, system=system)
        response = self._call(self._messages(prompt, system), response_format=schema)
        return self.parse_reply(self._content(response), schema)

    def _call(self, messages: list[dict[str, str]], **overrides: Any) -> Any:
        try:
            return self._litellm().completion(
                model=self.model_name, messages=messages, **self._params, **overrides
            )
        except ImportError:
            raise  # a missing extra is an install problem, not a failed call
        except Exception as exc:
            raise LlmInvocationError(
                f"litellm call to {self.model_name!r} failed: {exc}"
            ) from exc

    def _supports_schema(self) -> bool:
        """Does this model constrain output to a JSON Schema natively?

        An unrecognised model is not an error here — it only selects the
        inherited prompt-and-parse path.

        Returns:
            True when litellm reports native response-schema support.

        Raises:
            ImportError: If litellm is not installed.
        """
        try:
            return bool(self._litellm().supports_response_schema(model=self.model_name))
        except ImportError:
            raise  # a missing extra is an install problem, not a failed call
        except Exception:
            return False

    @staticmethod
    def _litellm() -> Any:
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        return litellm

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LlmInvocationError(
                f"unexpected litellm response: {response!r}"
            ) from exc
        if not isinstance(content, str) or not content:
            raise LlmInvocationError("litellm returned an empty completion")
        return content
