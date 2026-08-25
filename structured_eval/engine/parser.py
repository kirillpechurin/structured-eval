"""Coercion of raw sample input into Python values, without raising."""

from __future__ import annotations

from typing import Any

from structured_eval.formats.base import ParseError
from structured_eval.formats.json_parser import JsonParser


class Parser:
    """Coerces raw sample input into Python values without raising.

    Already-structured input (dict, list, `None`, scalars) passes through. A
    string is parsed as JSON; if that fails it is retried as YAML (when PyYAML
    is installed) and accepted only when it yields a dict or a list.

    A malformed model output is data, not an exception: `parse` reports the
    failure in its return value so the engine can surface it as a
    `parse_error` on the `EvalReport` instead of blowing up mid-batch.

    Example:
        >>> from structured_eval.engine import Parser
        >>> Parser().parse('{"status": "paid"}')
        ({'status': 'paid'}, None)
        >>> Parser().parse({"status": "paid"})       # passes through
        ({'status': 'paid'}, None)
        >>> value, error = Parser().parse("{oops")
        >>> value is None and error is not None
        True
    """

    def __init__(self) -> None:
        """Create a parser over the JSON backend; YAML is reached lazily."""
        self._json = JsonParser()

    def parse(self, raw: Any) -> tuple[Any, str | None]:
        """Coerce one raw sample value into a Python value.

        Args:
            raw: The value as given on the sample — already-structured data, or
                a string holding a JSON or YAML document.

        Returns:
            `(value, None)` on success, `(None, message)` when the string
            parsed as neither JSON nor YAML.

        Example:
            >>> from structured_eval.engine import Parser
            >>> Parser().parse('[{"total": 10}]')
            ([{'total': 10}], None)
            >>> Parser().parse(42)                   # not a string, passed through
            (42, None)
            >>> value, error = Parser().parse("not a document")
            >>> value, error.startswith("Invalid JSON")
            (None, True)
        """
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
        """Parse `text` as YAML, returning a dict/list or None on any failure."""
        from structured_eval.formats.yaml_parser import YamlParser

        try:
            value = YamlParser().parse(text)
        except (ParseError, ImportError):
            return None
        return value if isinstance(value, (dict, list)) else None
