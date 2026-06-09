"""Turkish-tuned quality-filter parameters.

Generic Gopher/C4 quality filters were calibrated on English. Two of their checks are
distorted by Turkish morphology and MUST be retuned, or they silently discard good text:

1. Stop-word presence. Gopher rejects documents with fewer than N stop words and uses a
   stop-word ratio. These require a *Turkish* stop-word list — an English list rejects
   legitimate Turkish prose wholesale.

2. Mean word length. Turkish is agglutinative: one stem yields long surface forms
   (``ev -> evler -> evlerimizden -> evlerimizdenmiş``). The English 3.0-10.0 character
   band is too tight and rejects normal Turkish. We widen the upper bound.

The values here are now DERIVED, not hand-set, following FineWeb-2's method (where Turkish
was a tuned "canary" language): :data:`TURKISH_STOPWORDS` and the thresholds are read from a
clean Turkish reference corpus (Turkish Wikipedia) via ``scripts/calibrate_filters.py``,
which writes artifacts into ``turkish_corpus/data/``. When those artifacts are absent (e.g.
a fresh checkout before calibration is run), this module transparently falls back to the
curated :data:`_CURATED_STOPWORDS` and the dataclass defaults — so it works either way.

Pure data + a dataclass; no datatrove import, so this stays unit-testable. ``pipeline.py``
feeds these into datatrove's filter blocks.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass, field

__all__ = ["TURKISH_STOPWORDS", "TurkishQualityThresholds"]

# Artifact file names produced by scripts/calibrate_filters.py (kept in sync with
# turkish_corpus.calibration; duplicated as literals here to avoid an import cycle).
_DATA_PACKAGE = "turkish_corpus"
_DATA_SUBDIR = "data"
_STOPWORDS_FILENAME = "turkish_stopwords.txt"
_THRESHOLDS_FILENAME = "turkish_thresholds.json"

# Curated high-frequency Turkish function words. This is the FALLBACK used when no derived
# stopword artifact is present; the calibration step replaces it with a frequency-derived list.
_CURATED_STOPWORDS: frozenset[str] = frozenset(
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


def _data_resource(filename: str) -> importlib.resources.abc.Traversable:
    """Return the importlib.resources handle for a calibration artifact under ``data/``."""
    return importlib.resources.files(_DATA_PACKAGE) / _DATA_SUBDIR / filename


def _load_stopwords() -> frozenset[str]:
    """Load the derived stopword artifact if present and non-empty, else the curated set.

    Reading via ``importlib.resources`` works whether the package is installed or run from a
    source checkout, and whether or not the artifact file exists yet (pre-calibration).
    """
    try:
        text = _data_resource(_STOPWORDS_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return _CURATED_STOPWORDS
    words = frozenset(line.strip() for line in text.splitlines() if line.strip())
    return words or _CURATED_STOPWORDS


# Resolved at import time: derived list after calibration, curated list before. Callers and
# existing tests keep using TURKISH_STOPWORDS without caring which source produced it.
TURKISH_STOPWORDS: frozenset[str] = _load_stopwords()


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

    def __post_init__(self) -> None:
        """Fail fast on inverted/negative bands so a bad calibration can't silently ship.

        A derived band where min > max would reject everything downstream; catching it here
        (rather than after a multi-hour pipeline run) is cheap and unambiguous.
        """
        if self.min_doc_words > self.max_doc_words:
            raise ValueError(
                f"min_doc_words ({self.min_doc_words}) > max_doc_words ({self.max_doc_words})"
            )
        if self.min_avg_word_length > self.max_avg_word_length:
            raise ValueError(
                f"min_avg_word_length ({self.min_avg_word_length}) > "
                f"max_avg_word_length ({self.max_avg_word_length})"
            )
        for name in ("max_symbol_word_ratio", "max_bullet_lines_ratio",
                     "max_ellipsis_lines_ratio", "max_non_alpha_words_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @classmethod
    def calibrated(cls) -> TurkishQualityThresholds:
        """Build thresholds from the derived JSON artifact, falling back to defaults.

        Loads ``turkish_corpus/data/turkish_thresholds.json`` (the numeric kwargs written by
        ``scripts/calibrate_filters.py``) and merges it OVER the dataclass defaults, so a
        partial artifact (only some fields) still yields a complete, valid object. ``stop_words``
        always comes from :data:`TURKISH_STOPWORDS` (itself derived-or-curated). When the
        artifact is absent — fresh checkout before calibration — returns plain defaults.
        """
        try:
            raw = _data_resource(_THRESHOLDS_FILENAME).read_text(encoding="utf-8")
            overrides = json.loads(raw)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            # A missing artifact (fresh checkout) OR a corrupt/unparseable one must fail SAFE
            # to defaults rather than crash at import time (config defaults to calibrated()).
            return cls()
        if not isinstance(overrides, dict):
            return cls()
        # Only accept known numeric fields; ignore stop_words even if present in the JSON.
        allowed = {
            "min_doc_words", "max_doc_words", "min_avg_word_length", "max_avg_word_length",
            "max_symbol_word_ratio", "max_bullet_lines_ratio", "max_ellipsis_lines_ratio",
            "max_non_alpha_words_ratio", "min_stop_words",
        }
        # Type-guard each merged value: a wrong-typed field (e.g. a string) must not reach
        # __post_init__ / downstream filters. Drop non-numeric values (bools excluded).
        kwargs = {
            k: v
            for k, v in overrides.items()
            if k in allowed and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return cls(stop_words=TURKISH_STOPWORDS, **kwargs)

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
