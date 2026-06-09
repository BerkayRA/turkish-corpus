"""Custom datatrove pipeline blocks wrapping the Turkish-specific logic.

Importing this module requires the ``pipeline`` extra (datatrove). The pure logic lives
in :mod:`turkish_corpus.normalization` and :mod:`turkish_corpus.pii` and is tested
independently; these thin wrappers adapt it to datatrove's ``PipelineStep`` interface and
emit per-type statistics so a run is auditable (important for KVKK provenance).
"""

from __future__ import annotations

import hashlib

try:
    from datatrove.data import DocumentsPipeline
    from datatrove.pipeline.base import PipelineStep
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "turkish_corpus.blocks requires datatrove. Install with `uv sync --extra pipeline`."
    ) from exc

from .normalization import normalize_text
from .pii import redact_turkish_pii

__all__ = ["TurkishNormalizer", "TurkishPIIRedactor", "TurkishTokensCounter"]


def _stable_unit(s: str) -> float:
    """Map a string deterministically to ``[0, 1)`` via md5.

    Builtin ``hash()`` is salted per-process (``PYTHONHASHSEED``), so it can't drive sampling
    that must select the *same* documents across runs/workers. md5 is stable everywhere; we
    only need a uniform spread, not cryptographic strength.
    """
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(digest, 16) / 2**128


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


class TurkishTokensCounter(PipelineStep):
    """Count tokens in-pipeline with *any* of our counters, not just HF tokenizers.

    datatrove's built-in ``TokensCounter`` only accepts an HF ``tokenizer.json`` / Hub id, so
    it can't count SentencePiece ``.model`` files or — the one that matters — the custom
    morpheme-aware BPE (``morpheme_bpe_<V>.json``), which is not exportable to a faithful HF
    file because it needs the ``tr_api`` morphological analyzer per word at inference. This
    block bridges that gap: it builds the matching
    :class:`turkish_corpus.tokenizer.TokenCounter` and records, per document,
    ``metadata["token_count"]`` plus a cumulative ``tokens`` run-stat (and ``counted_docs``).

    morpheme-BPE counting needs the ``turkish-tokenizer`` repo on ``sys.path`` via
    ``TURKISH_TOKENIZER_PATH`` (or an explicit ``repo_path``).

    SPEED CAVEAT
    ------------
    The morpheme-BPE path runs the pure-Python analyzer at ~600 words/s, far too slow to count
    every document of a billion-token corpus. For large runs set ``sample_rate < 1`` to
    tokenize a deterministic sample and ESTIMATE the corpus total as
    ``tokens_stat * (1 / sample_rate)``. HF and SentencePiece counters are fast enough to leave
    ``sample_rate == 1.0`` (count everything).

    The counter is built LAZILY in :meth:`run` so the heavy ``tr_api`` / HF import happens in
    the datatrove worker process, not at pipeline-construction time.
    """

    name = "🔢 Turkish Tokens Counter"
    type = "📊 - STATS"

    def __init__(
        self,
        tokenizer_spec: str,
        *,
        repo_path: str | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate!r}")
        self.tokenizer_spec = tokenizer_spec
        self.repo_path = repo_path
        self.sample_rate = sample_rate
        self._counter = None  # built lazily in run()

    def _build_counter(self):
        # Lazy imports: keep pipeline construction (and `import blocks`) free of tr_api/HF.
        from .tokenizer import is_morpheme_bpe_spec, load_token_counter  # noqa: PLC0415

        if is_morpheme_bpe_spec(self.tokenizer_spec):
            # Route through MorphemeBPETokenCounter directly so repo_path reaches it
            # (load_token_counter forwards **kwargs, but we want it explicit here).
            from .morpheme_bpe import MorphemeBPETokenCounter  # noqa: PLC0415

            return MorphemeBPETokenCounter(self.tokenizer_spec, repo_path=self.repo_path)
        return load_token_counter(self.tokenizer_spec)

    def _should_count(self, doc_id: str) -> bool:
        return self.sample_rate >= 1.0 or _stable_unit(doc_id) < self.sample_rate

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if self._counter is None:
            self._counter = self._build_counter()
        for doc in data:
            if self._should_count(doc.id):
                with self.track_time():
                    n = self._counter.count(doc.text)
                doc.metadata["token_count"] = n
                self.stat_update("tokens", value=n)
                self.stat_update("counted_docs")
            self.stat_update("documents")
            yield doc
