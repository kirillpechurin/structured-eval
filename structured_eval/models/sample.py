"""`Sample` — one document to evaluate, with its reference and optional source."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Sample(BaseModel):
    """One document to evaluate.

    Wrapping in `Sample` removes the ambiguity of a bare `list`: a list passed as
    `actual` is a single document whose root is an array, whereas `list[Sample]`
    is a batch of documents.

    Attributes:
        actual: The document under evaluation, parsed or as raw text.
        expected: The reference document to score it against.
        source: The original text a faithfulness metric grades the values against.
        id: Identifier for this sample in a `BatchEvalReport`.
    """

    actual: dict[str, Any] | list[Any] | str
    expected: dict[str, Any] | list[Any] | str | None = None
    source: str | None = None
    id: str | None = None
