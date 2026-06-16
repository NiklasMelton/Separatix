"""Multilabel diagnostic helpers for separatix."""

from separatix.multilabel.pipeline import fit_multilabel
from separatix.multilabel.validation import (
    ValidatedMultilabelInput,
    is_multilabel_indicator,
    validate_multilabel_inputs,
)

__all__ = [
    "ValidatedMultilabelInput",
    "fit_multilabel",
    "is_multilabel_indicator",
    "validate_multilabel_inputs",
]
