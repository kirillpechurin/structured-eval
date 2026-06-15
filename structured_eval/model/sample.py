from __future__ import annotations

from pydantic import BaseModel


class Sample(BaseModel):
    """One document to evaluate.

    Wrapping in ``Sample`` removes the ambiguity of a bare ``list``: a list
    passed as ``actual`` is a single document whose root is an array, whereas
    ``list[Sample]`` is a batch of documents.
    """

    actual: dict | list | str
    expected: dict | list | str | None = None
    source: str | None = None  # original text, for Faithfulness
    id: str | None = None  # identifier in a BatchEvalReport
