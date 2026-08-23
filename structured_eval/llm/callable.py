"""The smallest client there is: a function that already *is* ``generate``."""

from __future__ import annotations

from typing import Protocol

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmInvocationError


class GenerateFn(Protocol):
    """`LlmClient.generate` as a plain function — what `CallableClient` takes."""

    def __call__(self, prompt: str, *, system: str | None = None) -> str:
        """Answer `prompt` with the model's reply text."""
        ...


class CallableClient(LlmClient):
    """Wraps a function with `generate`'s own signature.

    Useful for tests, local pipelines, and any provider reachable through one
    function. The call is forwarded verbatim — the function owns what to do with
    `system`, exactly as a subclass would. Schema support comes from the base
    class: implementing this one function is enough for both methods.

    Example:
        >>> from structured_eval.llm import CallableClient
        >>> def my_model(prompt: str, *, system: str | None = None) -> str:
        ...     return my_sdk.complete(system=system, user=prompt).text
        >>> client = CallableClient(my_model)
        >>> client.model_name           # the function's own name, by default
        'my_model'
        >>> client.generate("Is the summary faithful?")  # doctest: +SKIP
        'Yes — the total and the date both appear in the source.'
    """

    def __init__(self, fn: GenerateFn, *, model_name: str = "") -> None:
        """Adopt `fn` as this client's `generate`.

        Args:
            fn: The function to call, with `generate`'s own signature.
            model_name: What to record as the model; defaults to the function's
                `__name__`.
        """
        self._fn = fn
        self.model_name = model_name or getattr(fn, "__name__", "callable")

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Call the wrapped function, as the provider call it stands for.

        Args:
            prompt: The user turn.
            system: The system turn, passed through untouched.

        Returns:
            Whatever the function returned.

        Raises:
            LlmInvocationError: If the function raised; the original exception
                is kept as `__cause__`.
        """
        try:
            return self._fn(prompt, system=system)
        except Exception as exc:
            raise LlmInvocationError(f"callable client failed: {exc}") from exc
