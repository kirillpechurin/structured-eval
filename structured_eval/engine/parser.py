from __future__ import annotations

from typing import Any

from structured_eval.formats.base import ParseError
from structured_eval.formats.json_parser import JsonParser


class Parser:
    """Coerces raw sample input into Python values without raising.

    Already-structured input (dict/list/None/scalars) passes through. A string
    is parsed as JSON; if that fails it is retried as YAML (when PyYAML is
    installed) and accepted only when it yields a dict or list. ``parse`` returns
    ``(value, None)`` on success or ``(None, message)`` on failure, so the engine
    can surface a parse error in the ``EvalReport`` rather than blowing up.
    """

    def __init__(self) -> None:
        self._json = JsonParser()

    def parse(self, raw: Any) -> tuple[Any, str | None]:
        if not isinstance(raw, str):
            return raw, None
        try:
            return self._json.parse(raw), None
        except ParseError as json_error:
            value = self._try_yaml(raw)
            if value is not None:
                return value, None
            return None, str(json_error)

    @staticmethod
    def _try_yaml(text: str) -> Any | None:
        """Parse ``text`` as YAML, returning a dict/list or None on any failure."""
        from structured_eval.formats.yaml_parser import YamlParser

        try:
            value = YamlParser().parse(text)
        except (ParseError, ImportError):
            return None
        return value if isinstance(value, (dict, list)) else None
