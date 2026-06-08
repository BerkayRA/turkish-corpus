"""Turkish-tuned quality-filter parameters.

Generic Gopher/C4 quality filters were calibrated on English. Two of their checks are
distorted by Turkish morphology and MUST be retuned, or they silently discard good text:

1. Stop-word presence. Gopher rejects documents with fewer than N stop words and uses a
   stop-word ratio. These require a *Turkish* stop-word list — an English list rejects
   legitimate Turkish prose wholesale.

2. Mean word length. Turkish is agglutinative: one stem yields long surface forms
   (``ev -> evler -> evlerimizden -> evlerimizdenmiş``). The English 3.0-10.0 character
   band is too tight and rejects normal Turkish. We widen the upper bound.

The values below are sensible defaults. The rigorous approach (FineWeb-2's method, where
Turkish was a tuned "canary" language) is to DERIVE thresholds from a clean Turkish
reference corpus (Turkish Wikipedia) per the "10Tail"/"Quantile"/"MeanStd" strategies.
:data:`TURKISH_STOPWORDS` is a curated starter list; replace it with a frequency-derived
list when you build the reference-corpus calibration step.

Pure data + a dataclass; no datatrove import, so this stays unit-testable. ``pipeline.py``
feeds these into datatrove's filter blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TURKISH_STOPWORDS", "TurkishQualityThresholds"]

# Curated high-frequency Turkish function words. Intentionally small and conservative;
# derive a fuller list from word-frequency over Turkish Wikipedia for production.
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "acaba", "ama", "ancak", "artık", "asla", "aslında", "az", "bana", "bazen",
        "bazı", "belki", "ben", "beni", "benim", "beş", "bile", "bir", "birçok",
        "biri", "birkaç", "birşey", "biz", "bize", "bizi", "bizim", "böyle", "böylece",
        "bu", "buna", "bunda", "bundan", "bunu", "bunun", "burada", "bütün", "çok",
        "çünkü", "da", "daha", "dahi", "de", "değil", "diğer", "diye", "doğru", "elbette",
        "en", "fakat", "gibi", "göre", "hâlâ", "hangi", "hatta", "hem", "henüz", "hep",
        "hepsi", "her", "herkes", "hiç", "hiçbir", "için", "içinde", "ile", "ilgili",
        "ise", "işte", "kadar", "karşı", "kendi", "ki", "kim", "madem", "mı", "mi",
        "mu", "mü", "nasıl", "ne", "neden", "nedenle", "nerede", "nereye", "niçin",
        "niye", "o", "olan", "olarak", "oldu", "olduğu", "olur", "ona", "onlar",
        "onların", "onu", "onun", "orada", "öyle", "pek", "rağmen", "sana", "sanki",
        "sen", "senin", "siz", "sizin", "şey", "şu", "şöyle", "tüm", "üzere", "var",
        "ve", "veya", "ya", "yani", "yine", "yoksa",
    }
)


@dataclass
class TurkishQualityThresholds:
    """Gopher-style quality thresholds adjusted for Turkish.

    Field names mirror datatrove's ``GopherQualityFilter`` so they map straight through.
    """

    min_doc_words: int = 50
    max_doc_words: int = 100_000
    # Widened upper bound (default English is 10.0) for agglutinative word lengths.
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 12.0
    max_symbol_word_ratio: float = 0.10
    max_bullet_lines_ratio: float = 0.90
    max_ellipsis_lines_ratio: float = 0.30
    max_non_alpha_words_ratio: float = 0.70
    min_stop_words: int = 2
    stop_words: frozenset[str] = field(default_factory=lambda: TURKISH_STOPWORDS)

    def as_gopher_kwargs(self) -> dict[str, object]:
        """Return kwargs for ``datatrove.pipeline.filters.GopherQualityFilter``."""
        return {
            "min_doc_words": self.min_doc_words,
            "max_doc_words": self.max_doc_words,
            "min_avg_word_length": self.min_avg_word_length,
            "max_avg_word_length": self.max_avg_word_length,
            "max_symbol_word_ratio": self.max_symbol_word_ratio,
            "max_bullet_lines_ratio": self.max_bullet_lines_ratio,
            "max_ellipsis_lines_ratio": self.max_ellipsis_lines_ratio,
            "max_non_alpha_words_ratio": self.max_non_alpha_words_ratio,
            "min_stop_words": self.min_stop_words,
            "stop_words": list(self.stop_words),
        }
