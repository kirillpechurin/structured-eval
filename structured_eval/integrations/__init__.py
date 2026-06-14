"""Adapters that plug structured-eval into host eval frameworks.

The core (``evaluate`` → ``EvalReport``) is framework-agnostic; each adapter
lazily imports its host library so ``import structured_eval`` never requires
deepeval/langsmith. Install with the matching extra (``structured-eval[deepeval]``
or ``[langsmith]``).
"""

from structured_eval.integrations._adapter import reason_text, verdict

__all__ = ["verdict", "reason_text"]
