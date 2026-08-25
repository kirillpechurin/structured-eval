"""Array alignment — pairing actual list items with expected ones.

An aligner answers a single question: which actual element does each expected
element correspond to? Array precision/recall/F1, per-element scores and
cardinality all read that pairing:

- `ArrayAligner` — the interface: implement `align` and you have a strategy.
- `ByIndexAligner` — pairs by position.
- `ByKeyAligner` — pairs on a matching key, greedily best-first.
- `HungarianAligner` — optimal one-to-one assignment, behind the `align` extra.
- `make_aligner` — builds the strategy an `ArrayFieldConfig` names.
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
