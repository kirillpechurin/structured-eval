"""Array alignment — pairing actual list items with expected ones.

An aligner answers a single question: which actual element does each expected
element correspond to? Everything downstream reads that pairing — array
precision/recall/F1, per-element scores, cardinality. `make_aligner` builds the
strategy an `ArrayFieldConfig` names; `by_index`, `by_key` and `hungarian`
implement the three of them.
"""

from structured_eval.alignment.base import ArrayAligner, key_value, keyable
from structured_eval.alignment.by_index import ByIndexAligner
from structured_eval.alignment.by_key import ByKeyAligner
from structured_eval.alignment.factory import make_aligner
from structured_eval.alignment.hungarian import HungarianAligner, Scorer

__all__ = [
    "ArrayAligner",
    "ByIndexAligner",
    "ByKeyAligner",
    "HungarianAligner",
    "Scorer",
    "key_value",
    "keyable",
    "make_aligner",
]
