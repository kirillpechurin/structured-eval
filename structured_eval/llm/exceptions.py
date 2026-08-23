"""The error hierarchy of the LLM layer, rooted at `LlmError`."""


class LlmError(RuntimeError):
    """Base for every error raised by this layer — catch this to catch all.

    Example:
        >>> from structured_eval.llm import LiteLlmClient, LlmError
        >>> client = LiteLlmClient("openai/gpt-4o", timeout=10)
        >>> try:  # doctest: +SKIP
        ...     client.generate("Is the summary faithful?")
        ... except LlmError as exc:  # a failed call and an unreadable reply
        ...     type(exc).__name__
        'LlmInvocationError'
    """


class LlmInvocationError(LlmError):
    """The underlying provider call failed.

    The provider's own exception (timeout, auth, rate limit, transport) is kept
    as `__cause__`. This layer owns no transport and adds no retry policy of its
    own: retries, timeouts, and backoff belong to the client you inject.

    Example:
        >>> from structured_eval.llm import LiteLlmClient, LlmInvocationError
        >>> client = LiteLlmClient("openai/gpt-4o", timeout=10)
        >>> try:  # doctest: +SKIP
        ...     client.generate("Is the summary faithful?")
        ... except LlmInvocationError as exc:
        ...     type(exc.__cause__).__name__  # the provider's own error
        'APIConnectionError'
    """


class LlmResponseFormatError(LlmError):
    """The reply could not be read as the requested schema.

    Raised for a reply holding no JSON object at all, and for one whose JSON
    does not validate — the message quotes the reply's opening characters, so
    a failing judge shows what the model actually said.

    Example:
        >>> from structured_eval.llm import LiteLlmClient, LlmResponseFormatError
        >>> from structured_eval.models import JudgeVerdict
        >>> client = LiteLlmClient("openai/gpt-4o")
        >>> try:  # the model answered in prose instead of JSON
        ...     client.parse_reply("Hard to say, honestly.", JudgeVerdict)
        ... except LlmResponseFormatError as exc:
        ...     str(exc)
        "no JSON object in model reply: 'Hard to say, honestly.'"
    """
