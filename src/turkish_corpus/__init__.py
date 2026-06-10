"""turkish-corpus — pipeline + crawler for a high-quality ~1B-token Turkish LLM corpus.

The top-level package exposes only the pure, dependency-free Turkish-specific helpers so
``import turkish_corpus`` works without the heavy ``pipeline`` extra. The datatrove
integration lives in :mod:`turkish_corpus.blocks` and :mod:`turkish_corpus.pipeline` and
is imported on demand.
"""

from __future__ import annotations

from .config import PipelineConfig, default_hplt_config
from .normalization import (
    nfc,
    normalize_for_dedup,
    normalize_text,
    turkish_lower,
    turkish_title,
    turkish_upper,
)
from .pii import RedactionResult, redact_turkish_pii, validate_tc_kimlik, validate_tr_iban
from .tokenizer import load_token_counter

__all__ = [
    "PipelineConfig",
    "default_hplt_config",
    "nfc",
    "turkish_lower",
    "turkish_upper",
    "turkish_title",
    "normalize_for_dedup",
    "normalize_text",
    "redact_turkish_pii",
    "validate_tc_kimlik",
    "validate_tr_iban",
    "RedactionResult",
    "load_token_counter",
]

__version__ = "0.1.0"
