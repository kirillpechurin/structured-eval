"""Adapters that plug structured-eval into host eval frameworks.

The core (`evaluate` → `EvalReport`) is framework-agnostic, and each adapter
lives in its own module importing its host library, so `import structured_eval`
never requires deepeval or langsmith. Install with the matching extra —
`structured-eval[deepeval]` or `structured-eval[langsmith]`.
"""

from structured_eval.integrations._adapter import reason_text, verdict

__all__ = ["reason_text", "verdict"]
