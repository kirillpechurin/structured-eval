class LlmError(RuntimeError):
    """Base for every error raised by this layer — catch this to catch all."""


class LlmInvocationError(LlmError):
    """The underlying provider call failed.

    The provider's own exception (timeout, auth, rate limit, transport) is kept
    as ``__cause__``. This layer owns no transport and adds no retry policy of
    its own: retries, timeouts, and backoff belong to the client you inject.
    """


class LlmResponseFormatError(LlmError):
    """The reply could not be read as the requested schema."""
