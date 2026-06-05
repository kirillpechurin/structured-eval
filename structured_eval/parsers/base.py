from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ParseError(ValueError):
    """Raised when input text cannot be parsed into a structured value."""


@runtime_checkable
class Parser(Protocol):
    """Parse a text string into a Python object.

    Implementations must raise ParseError on malformed input.
    Return type is Any because parsers may produce dict, list, or scalar
    depending on the input (e.g. JSONL returns an iterator of dicts).
    """

    def parse(self, text: str) -> Any: ...
