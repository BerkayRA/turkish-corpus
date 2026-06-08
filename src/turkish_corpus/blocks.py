"""Custom datatrove pipeline blocks wrapping the Turkish-specific logic.

Importing this module requires the ``pipeline`` extra (datatrove). The pure logic lives
in :mod:`turkish_corpus.normalization` and :mod:`turkish_corpus.pii` and is tested
independently; these thin wrappers adapt it to datatrove's ``PipelineStep`` interface and
emit per-type statistics so a run is auditable (important for KVKK provenance).
"""

from __future__ import annotations

try:
    from datatrove.data import DocumentsPipeline
    from datatrove.pipeline.base import PipelineStep
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "turkish_corpus.blocks requires datatrove. Install with `uv sync --extra pipeline`."
    ) from exc

from .normalization import normalize_text
from .pii import redact_turkish_pii

__all__ = ["TurkishNormalizer", "TurkishPIIRedactor"]


class TurkishNormalizer(PipelineStep):
    """NFC + control/zero-width strip + whitespace cleanup, preserving case."""

    name = "🧹 Turkish Normalizer"
    type = "🔧 - FORMAT"

    def __init__(self, *, fix_mojibake: bool = False) -> None:
        super().__init__()
        self.fix_mojibake = fix_mojibake

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            with self.track_time():
                doc.text = normalize_text(doc.text, fix_mojibake=self.fix_mojibake)
            self.stat_update("normalized")
            yield doc


class TurkishPIIRedactor(PipelineStep):
    """Redact Turkish-specific PII (T.C. Kimlik, phone, IBAN, optionally plate)."""

    name = "🛡 Turkish PII Redactor"
    type = "🔧 - FORMAT"

    def __init__(
        self,
        *,
        redact_tc: bool = True,
        redact_phone: bool = True,
        redact_iban: bool = True,
        redact_plate: bool = False,
    ) -> None:
        super().__init__()
        self.redact_tc = redact_tc
        self.redact_phone = redact_phone
        self.redact_iban = redact_iban
        self.redact_plate = redact_plate

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            with self.track_time():
                result = redact_turkish_pii(
                    doc.text,
                    redact_tc=self.redact_tc,
                    redact_phone=self.redact_phone,
                    redact_iban=self.redact_iban,
                    redact_plate=self.redact_plate,
                )
            doc.text = result.text
            for kind, n in result.counts.items():
                self.stat_update(f"pii_{kind}", value=n)
            self.stat_update("documents")
            yield doc
