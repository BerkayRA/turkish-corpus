"""Derive corpus-quality-filter parameters from a clean Turkish reference corpus.

This is the FineWeb-2 "canary language" method: rather than hand-setting the Turkish
stop-word list and the Gopher/quality thresholds (which is what :mod:`.filters` ships as a
curated *fallback*), we DERIVE them from a known-clean reference corpus — Turkish Wikipedia.

Why derive instead of hand-tune:

- A frequency-derived stop-word list reflects the actual function-word distribution of the
  language, not an English speaker's guess at it. Gopher's stop-word check then keys off
  the real high-frequency Turkish words.
- The Gopher metric bands (mean word length, doc length, symbol/non-alpha ratios) were
  calibrated on English. Turkish is agglutinative — surface forms run long — so the English
  bands silently discard good prose. Reading the bands off the *observed* per-document
  distribution of a clean corpus (quantiles, or mean ± k·std) gives language-correct cutoffs.

Design notes
------------
- This module is PURE: it accumulates plain-Python stats and derives values with stdlib only
  (no numpy — it is heavy/optional and a hand-rolled linear-interpolation quantile is enough
  for a one-shot calibration). ``datasets`` is not imported here; the CLI runner
  (``scripts/calibrate_filters.py``) feeds article texts in as a plain iterable, so this is
  fully unit-testable with synthetic text and no network.
- Casing uses :func:`turkish_corpus.normalization.turkish_lower` so the ı/İ distinction is
  preserved — folding it would merge distinct Turkish words in the frequency table.
- Artifacts are written as a plain stopwords text file + a thresholds JSON so they can be
  committed and diffed; :mod:`.filters` loads them via ``importlib.resources`` with the
  curated list as fallback.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .normalization import turkish_lower

if TYPE_CHECKING:  # pragma: no cover
    # Imported only for annotations; the real class lives in .filters and derive_thresholds
    # imports it lazily at call time to keep this module free of an import cycle.
    from .filters import TurkishQualityThresholds

__all__ = [
    "STOPWORDS_FILENAME",
    "THRESHOLDS_FILENAME",
    "REPORT_FILENAME",
    "CorpusStats",
    "derive_stopwords",
    "derive_thresholds",
    "calibrate_from_corpus",
    "write_artifacts",
    "load_stopwords",
    "load_thresholds",
]

STOPWORDS_FILENAME = "turkish_stopwords.txt"
THRESHOLDS_FILENAME = "turkish_thresholds.json"
REPORT_FILENAME = "calibration_report.json"

# Characters that count toward the Gopher symbol/word ratio. "#" plus both the precomposed
# ellipsis (U+2026) and its ASCII spelling, matching what datatrove's Gopher filter treats
# as "symbols" in this corpus context.
_SYMBOLS = ("#", "…", "...")


def _quantile(sorted_values: list[float], p: float) -> float:
    """Return the ``p``-quantile (0..1) of an ascending-sorted list via linear interpolation.

    Pure stdlib (no numpy). Uses the same "linear" / R-7 convention numpy defaults to:
    position ``p * (n - 1)`` interpolated between its neighbours. ``p`` is clamped to
    ``[0, 1]``; an empty input returns ``0.0`` so callers can floor it safely.
    """
    if not sorted_values:
        return 0.0
    if p <= 0.0:
        return sorted_values[0]
    if p >= 1.0:
        return sorted_values[-1]
    pos = p * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


@dataclass
class CorpusStats:
    """Accumulate word frequencies and per-document Gopher metrics over a corpus sample.

    Built incrementally via :meth:`add` so a streaming caller can feed documents one at a
    time. Per-document metrics are kept as plain lists (the sample is bounded by the caller's
    ``--limit``, so this stays small); :func:`derive_thresholds` reads quantiles off them.

    Only all-alphabetic tokens (``str.isalpha()``) feed the stop-word frequency table — that
    drops numbers, punctuation, and mixed junk so the derived stop-word list is real words.
    """

    word_freq: Counter[str] = field(default_factory=Counter)
    n_words: list[int] = field(default_factory=list)
    mean_word_length: list[float] = field(default_factory=list)
    symbol_word_ratio: list[float] = field(default_factory=list)
    non_alpha_words_ratio: list[float] = field(default_factory=list)
    n_docs: int = 0

    def add(self, text: str) -> None:
        """Fold one document into the running stats; empty/whitespace-only docs are skipped."""
        words = text.split()
        if not words:
            return
        self.n_docs += 1
        n = len(words)
        self.n_words.append(n)

        lowered = [turkish_lower(w) for w in words]
        self.mean_word_length.append(sum(len(w) for w in lowered) / n)

        # All-alphabetic, Turkish-lowercased tokens drive the stop-word frequency table.
        for w in lowered:
            if w.isalpha():
                self.word_freq[w] += 1

        symbols = sum(text.count(s) for s in _SYMBOLS)
        self.symbol_word_ratio.append(symbols / n)

        non_alpha = sum(1 for w in lowered if not any(ch.isalpha() for ch in w))
        self.non_alpha_words_ratio.append(non_alpha / n)


def derive_stopwords(stats: CorpusStats, *, top_n: int = 175, min_word_len: int = 1) -> list[str]:
    """Return the ``top_n`` most frequent alphabetic words, most-frequent first.

    Operates purely on the accumulated frequency :class:`~collections.Counter`. Words are
    already Turkish-lowercased. ``min_word_len`` drops very short tokens if you want to
    exclude single characters; the default keeps everything length >= 1.
    """
    candidates = (w for w, _ in stats.word_freq.most_common() if len(w) >= min_word_len)
    out: list[str] = []
    for word in candidates:
        out.append(word)
        if len(out) >= top_n:
            break
    return out


def _band(samples: list[float], strategy: str, q: float) -> tuple[float, float]:
    """Return a (low, high) band over ``samples`` for the chosen ``strategy``.

    - ``quantile``: the ``q`` and ``1 - q`` quantiles.
    - ``10tail``: the 0.10 and 0.90 quantiles (a fixed, gentler tail trim).
    - ``meanstd``: mean ± 3·std, clamped to the observed [min, max].
    """
    ordered = sorted(samples)
    if not ordered:
        return 0.0, 0.0
    if strategy == "10tail":
        return _quantile(ordered, 0.10), _quantile(ordered, 0.90)
    if strategy == "meanstd":
        n = len(ordered)
        mean = sum(ordered) / n
        var = sum((x - mean) ** 2 for x in ordered) / n
        std = var**0.5
        low = max(mean - 3 * std, ordered[0])
        high = min(mean + 3 * std, ordered[-1])
        return low, high
    if strategy == "quantile":
        return _quantile(ordered, q), _quantile(ordered, 1.0 - q)
    raise ValueError(f"unknown strategy: {strategy!r} (expected quantile/10tail/meanstd)")


def derive_thresholds(
    stats: CorpusStats,
    *,
    strategy: str = "quantile",
    q: float = 0.005,
    stop_words: frozenset[str] | None = None,
) -> TurkishQualityThresholds:
    """Derive a :class:`~turkish_corpus.filters.TurkishQualityThresholds` from observed stats.

    Reads the per-document metric distributions off ``stats`` and turns them into Gopher
    bands per ``strategy`` (see :func:`_band`). Floors are applied so a *clean* corpus (which
    has almost no symbols / non-alpha noise) does not produce near-zero caps that would then
    reject ordinary noisier-but-fine web text downstream:

    - ``min_avg_word_length`` / ``max_avg_word_length`` from the mean-word-length band.
    - ``min_doc_words`` = max(10, low-band of n_words) — drops stubs.
    - ``max_doc_words`` is kept GENEROUS (floored at the 100_000 default, ceiled at
      1_000_000): the upper bound is deliberately NOT pulled down to a short-article
      reference's quantile, because the production blend includes long-form registers
      (academic theses, laws) that legitimately run tens of thousands of words. Calibrating
      it from encyclopedic Wikipedia would silently drop entire theses.
    - ``max_symbol_word_ratio`` = high band, floored at 0.05.
    - ``max_non_alpha_words_ratio`` = high band, floored at 0.2.
    - Line-based ratios (bullet/ellipsis) and ``min_stop_words`` are NOT calibrated here;
      they keep the dataclass defaults.

    The dataclass is imported lazily so this module does not hard-couple to :mod:`.filters`
    at import time (and the import stays cheap / cycle-free).
    """
    from .filters import TURKISH_STOPWORDS, TurkishQualityThresholds  # noqa: PLC0415

    min_awl, max_awl = _band(stats.mean_word_length, strategy, q)
    low_words, high_words = _band([float(n) for n in stats.n_words], strategy, q)
    _, high_symbol = _band(stats.symbol_word_ratio, strategy, q)
    _, high_non_alpha = _band(stats.non_alpha_words_ratio, strategy, q)

    return TurkishQualityThresholds(
        min_doc_words=max(10, round(low_words)),
        # Floor at the generous default so a short-article reference never truncates
        # long-form theses/laws; only grow if the corpus genuinely has longer docs.
        max_doc_words=min(max(round(high_words), 100_000), 1_000_000),
        min_avg_word_length=min_awl,
        max_avg_word_length=max_awl,
        max_symbol_word_ratio=max(high_symbol, 0.05),
        max_non_alpha_words_ratio=max(high_non_alpha, 0.2),
        stop_words=stop_words if stop_words is not None else TURKISH_STOPWORDS,
    )


def calibrate_from_corpus(
    texts: Iterable[str],
    *,
    top_n: int = 175,
    strategy: str = "quantile",
    q: float = 0.005,
) -> tuple[list[str], TurkishQualityThresholds, dict]:
    """Accumulate stats over ``texts`` and derive (stopwords, thresholds, report).

    The ``report`` dict captures provenance for logging and for ``calibration_report.json``:
    sample size, vocabulary size, the chosen threshold values, and the raw metric quantiles
    used to set the bands — so a future reviewer can see exactly what the corpus implied.
    """
    stats = CorpusStats()
    for text in texts:
        stats.add(text)

    stopwords = derive_stopwords(stats, top_n=top_n)
    thresholds = derive_thresholds(
        stats, strategy=strategy, q=q, stop_words=frozenset(stopwords)
    )

    report = {
        "strategy": strategy,
        "q": q,
        "top_n": top_n,
        "n_docs": stats.n_docs,
        "n_unique_words": len(stats.word_freq),
        "thresholds": _threshold_numbers(thresholds),
        "sample_quantiles": _sample_quantiles(stats, q),
    }
    return stopwords, thresholds, report


def _threshold_numbers(thresholds: TurkishQualityThresholds) -> dict:
    """Numeric threshold fields only (drops ``stop_words``) — the committable JSON kwargs."""
    kwargs = thresholds.as_gopher_kwargs()
    kwargs.pop("stop_words", None)
    return kwargs


def _sample_quantiles(stats: CorpusStats, q: float) -> dict:
    """Raw low/high quantiles of each metric, for provenance in the report."""
    return {
        metric: {
            "low": _quantile(sorted(values), q),
            "high": _quantile(sorted(values), 1.0 - q),
        }
        for metric, values in (
            ("mean_word_length", stats.mean_word_length),
            ("n_words", [float(n) for n in stats.n_words]),
            ("symbol_word_ratio", stats.symbol_word_ratio),
            ("non_alpha_words_ratio", stats.non_alpha_words_ratio),
        )
    }


def write_artifacts(
    stopwords: Iterable[str],
    thresholds: TurkishQualityThresholds,
    out_dir: str | Path,
    *,
    report: dict | None = None,
) -> None:
    """Write the derived stopwords, thresholds, and optional report under ``out_dir``.

    Stopwords go one-per-line (UTF-8); thresholds are the numeric kwargs as JSON (the
    ``stop_words`` set is intentionally NOT written — it lives in the stopwords file so the
    two artifacts stay single-sourced). Creates ``out_dir`` if missing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / STOPWORDS_FILENAME).write_text(
        "\n".join(stopwords) + "\n", encoding="utf-8"
    )
    (out / THRESHOLDS_FILENAME).write_text(
        json.dumps(_threshold_numbers(thresholds), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if report is not None:
        (out / REPORT_FILENAME).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def load_stopwords(path: str | Path) -> frozenset[str]:
    """Load a one-per-line stopwords file into a frozenset; blank lines are ignored."""
    text = Path(path).read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def load_thresholds(path: str | Path) -> dict:
    """Load the numeric threshold kwargs JSON back into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
