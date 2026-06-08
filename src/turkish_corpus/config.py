"""Configuration for the HPLT v2 Turkish cleaning pipeline.

Pure dataclasses with no datatrove import, so configs can be constructed, validated, and
tested anywhere. ``pipeline.py`` consumes these to build the actual datatrove steps.

Data anchor: HPLT v2 cleaned, Turkish subset ``tur_Latn`` (~51.7B tokens, CC0 license) —
chosen over FineWeb-2 for the cleanest licensing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .filters import TurkishQualityThresholds

__all__ = [
    "ReaderConfig",
    "MinhashParams",
    "PIIConfig",
    "TokenizerConfig",
    "PipelineConfig",
    "default_hplt_config",
]

# HPLT v2 cleaned on the HuggingFace Hub. The config name IS the language code.
HPLT_DATASET = "HPLT/HPLT2.0_cleaned"
HPLT_LANG = "tur_Latn"


@dataclass
class ReaderConfig:
    """Where the raw HPLT text comes from.

    Two modes:
    - ``source="hf"``: stream ``HPLT/HPLT2.0_cleaned`` config ``tur_Latn`` from the Hub.
    - ``source="jsonl"``: read pre-downloaded JSONL(.zst) shards from ``data_path``
      (recommended for the full production run on the Linux box — download once, reread).
    """

    source: Literal["hf", "jsonl"] = "hf"
    dataset: str = HPLT_DATASET
    language: str = HPLT_LANG
    split: str = "train"
    data_path: str | None = None  # used when source == "jsonl"
    text_key: str = "text"
    id_key: str | None = "id"
    streaming: bool = True
    limit: int = -1  # -1 = no limit; set small for local dev slices


@dataclass
class MinhashParams:
    """MinHash near-dedup configuration (FineWeb-2's Turkish-validated settings)."""

    num_buckets: int = 14
    hashes_per_bucket: int = 8
    n_grams: int = 5
    precision: int = 64  # hash precision bits


@dataclass
class PIIConfig:
    """Toggles for PII scrubbing (KVKK)."""

    redact_emails_and_ips: bool = True  # via datatrove PIIFormatter
    redact_tc_kimlik: bool = True
    redact_phone: bool = True
    redact_iban: bool = True
    redact_plate: bool = False


@dataclass
class TokenizerConfig:
    """Token counting for the final ``TokensCounter`` block.

    ``name_or_path=None`` skips token counting. Point this at the morphology-aware
    tokenizer's ``tokenizer.json`` when ready (see ``tokenizer.py``).
    """

    name_or_path: str | None = None


@dataclass
class LanguageConfig:
    """Secondary language-ID gate. HPLT is already ``tur_Latn``-tagged, so this catches
    residual contamination rather than doing primary routing."""

    enabled: bool = True
    languages: tuple[str, ...] = ("tr",)
    threshold: float = 0.65  # per-language; tune from the confidence distribution


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    reader: ReaderConfig = field(default_factory=ReaderConfig)
    quality: TurkishQualityThresholds = field(default_factory=TurkishQualityThresholds)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    minhash: MinhashParams = field(default_factory=MinhashParams)
    pii: PIIConfig = field(default_factory=PIIConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    output_path: str = "./output/hplt_tur"
    logging_dir: str = "./logs/hplt_tur"
    # Local executor parallelism. On the Linux box raise `tasks`/`workers`; a Slurm
    # variant is provided in pipeline.py for cluster scale.
    tasks: int = 4
    workers: int = -1  # -1 = as many as tasks
    fix_mojibake: bool = False  # HPLT is pre-cleaned; enable for fresh crawl data

    def validate(self) -> None:
        """Fail fast on inconsistent configuration."""
        if self.reader.source not in {"hf", "jsonl"}:
            raise ValueError(f"reader.source must be 'hf' or 'jsonl', got {self.reader.source!r}")
        if self.reader.source == "jsonl" and not self.reader.data_path:
            raise ValueError("reader.data_path is required when reader.source == 'jsonl'")
        if self.minhash.num_buckets < 1 or self.minhash.hashes_per_bucket < 1:
            raise ValueError("minhash buckets/hashes must be >= 1")
        if not (0.0 <= self.language.threshold <= 1.0):
            raise ValueError("language.threshold must be in [0, 1]")


def default_hplt_config(**overrides) -> PipelineConfig:
    """Convenience constructor for the default HPLT v2 Turkish pipeline."""
    cfg = PipelineConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"PipelineConfig has no field {key!r}")
        setattr(cfg, key, value)
    cfg.validate()
    return cfg
