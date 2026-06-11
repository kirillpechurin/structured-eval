from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Name → matcher class. Populated automatically as MatcherBase subclasses are
# declared; used by EvalConfig.from_yaml() to resolve string names (Stage 10).
_MATCHER_REGISTRY: dict[str, type] = {}


@runtime_checkable
class Matcher(Protocol):
    """Structural type for matchers.

    A matcher reports how similar two leaf values are on a 0.0–1.0 scale.
    The result is a *similarity*, not a final score: metrics interpret it.
    """

    name: str

    def similarity(self, actual: Any, expected: Any) -> float: ...


class MatcherBase:
    """Base class for built-in and user-defined matchers.

    Subclasses set a class-level ``name`` and implement ``similarity()``.
    Declaring a subclass registers it by name automatically — no manual
    registration is needed for custom matchers.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if n := getattr(cls, "name", None):
            _MATCHER_REGISTRY[n] = cls

    def similarity(self, actual: Any, expected: Any) -> float:  # pragma: no cover
        raise NotImplementedError


def get_matcher_class(name: str) -> type:
    """Resolve a matcher class by its ``name`` (e.g. ``"NUMERIC"``)."""
    if name not in _MATCHER_REGISTRY:
        raise KeyError(
            f"Unknown matcher: {name!r}. Known: {sorted(_MATCHER_REGISTRY)}"
        )
    return _MATCHER_REGISTRY[name]
