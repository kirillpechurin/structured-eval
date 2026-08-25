"""The `Parser` interface and the `ParseError` every implementation raises."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ParseError(ValueError):
    """Raised when input text cannot be parsed into a structured value.

    It subclasses `ValueError`, so code that already treats malformed input as a
    value problem keeps working without knowing about this package.

    Example:
        >>> from structured_eval.formats import JsonParser, ParseError
        >>> try:
        ...     JsonParser().parse("not json")
        ... except ParseError as exc:
        ...     str(exc).startswith("Invalid JSON:")
        True
    """


@runtime_checkable
class Parser(Protocol):
    """Parse a text string into a Python object.

    Implementations must raise `ParseError` on malformed input. The return type
    is `Any` because a parser may produce a dict, a list or a scalar — `JsonlParser`
    returns an iterator of values.

    Example:
        >>> from structured_eval.formats import Parser
        >>> class CsvLineParser:
        ...     def parse(self, text: str) -> list[str]:
        ...         return text.split(",")
        >>> isinstance(CsvLineParser(), Parser)      # structural, no subclassing
        True
        >>> CsvLineParser().parse("acme,42")
        ['acme', '42']
    """

    def parse(self, text: str) -> Any:
        """Read `text` as a structured value, or raise `ParseError`."""
        ...
