"""Tests for the Turkish-Wikipedia-driven filter calibration.

These pin the pure derivation logic (quantiles, stat accumulation, stopword/threshold
derivation) and the artifact round-trip — all with synthetic Turkish text, no network.
"""

import pytest

from turkish_corpus.calibration import (
    STOPWORDS_FILENAME,
    THRESHOLDS_FILENAME,
    CorpusStats,
    _quantile,
    calibrate_from_corpus,
    derive_stopwords,
    derive_thresholds,
    load_stopwords,
    load_thresholds,
    write_artifacts,
)
from turkish_corpus.filters import TurkishQualityThresholds


class TestQuantile:
    def test_endpoints_return_min_and_max(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert _quantile(values, 0.0) == 1.0
        assert _quantile(values, 1.0) == 4.0

    def test_median_interpolates(self):
        # Position 0.5*(4-1)=1.5 -> halfway between values[1]=2 and values[2]=3.
        assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

    def test_empty_returns_zero(self):
        assert _quantile([], 0.5) == 0.0

    def test_clamps_out_of_range_p(self):
        assert _quantile([5.0, 10.0], -1.0) == 5.0
        assert _quantile([5.0, 10.0], 2.0) == 10.0


class TestCorpusStats:
    def test_add_accumulates_word_frequency_turkish_lowercased(self):
        stats = CorpusStats()
        stats.add("Ve bir VE bir Işık")  # "Işık" -> "ışık" under Turkish casing
        assert stats.word_freq["ve"] == 2
        assert stats.word_freq["bir"] == 2
        assert stats.word_freq["ışık"] == 1
        assert "Işık" not in stats.word_freq

    def test_add_records_per_doc_metrics(self):
        stats = CorpusStats()
        stats.add("ev evler evlerimiz")
        assert stats.n_docs == 1
        assert stats.n_words == [3]
        assert stats.mean_word_length[0] == (2 + 5 + 9) / 3

    def test_add_skips_empty_documents(self):
        stats = CorpusStats()
        stats.add("   ")
        stats.add("")
        assert stats.n_docs == 0
        assert stats.n_words == []

    def test_symbol_and_non_alpha_ratios(self):
        stats = CorpusStats()
        stats.add("bir # 123 ...")  # 4 words; symbols: one "#" + one "..." = 2
        assert stats.symbol_word_ratio[0] == 2 / 4
        # "#", "123", "..." have no alpha char -> 3 of 4 words; only "bir" is alphabetic.
        assert stats.non_alpha_words_ratio[0] == 3 / 4

    def test_only_alphabetic_tokens_enter_frequency_table(self):
        stats = CorpusStats()
        stats.add("bir 123 ve! ev3")
        assert stats.word_freq["bir"] == 1
        assert "123" not in stats.word_freq
        assert "ve!" not in stats.word_freq  # punctuation makes it non-alpha
        assert "ev3" not in stats.word_freq


class TestDeriveStopwords:
    def test_returns_most_frequent_words_first(self):
        stats = CorpusStats()
        # "ve" x3, "bir" x2, "ev" x1
        stats.add("Ve bir ve bir ev VE")
        words = derive_stopwords(stats, top_n=2)
        assert words == ["ve", "bir"]

    def test_respects_top_n_cap(self):
        stats = CorpusStats()
        stats.add("a a b b c c d d e e")
        assert len(derive_stopwords(stats, top_n=3)) == 3

    def test_min_word_len_filters_short(self):
        stats = CorpusStats()
        stats.add("a a bir bir")
        words = derive_stopwords(stats, top_n=10, min_word_len=2)
        assert "a" not in words
        assert "bir" in words


class TestDeriveThresholds:
    def _stats(self) -> CorpusStats:
        stats = CorpusStats()
        for _ in range(50):
            stats.add("ev evler evlerimiz kitap kitaplar okudum geldiler yazdılar")
        return stats

    def test_min_below_max_within_observed_range(self):
        stats = self._stats()
        t = derive_thresholds(stats, strategy="quantile", q=0.01)
        assert t.min_avg_word_length <= t.max_avg_word_length
        assert t.min_doc_words <= t.max_doc_words
        observed_min = min(stats.mean_word_length)
        observed_max = max(stats.mean_word_length)
        assert observed_min <= t.min_avg_word_length <= observed_max
        assert observed_min <= t.max_avg_word_length <= observed_max

    def test_floors_applied_for_clean_corpus(self):
        # Clean text has zero symbols/non-alpha; floors keep caps usable downstream.
        stats = self._stats()
        t = derive_thresholds(stats)
        assert t.max_symbol_word_ratio >= 0.05
        assert t.max_non_alpha_words_ratio >= 0.2

    def test_min_doc_words_floored_at_10(self):
        stats = CorpusStats()
        for _ in range(20):
            stats.add("bir iki üç")  # only 3 words each
        t = derive_thresholds(stats)
        assert t.min_doc_words == 10

    def test_max_doc_words_floored_at_1000(self):
        stats = self._stats()
        t = derive_thresholds(stats)
        assert t.max_doc_words >= 1000

    def test_stop_words_passed_through(self):
        stats = self._stats()
        custom = frozenset({"ve", "bir"})
        t = derive_thresholds(stats, stop_words=custom)
        assert t.stop_words == custom

    def test_strategy_variants_produce_valid_bands(self):
        stats = self._stats()
        for strategy in ("quantile", "10tail", "meanstd"):
            t = derive_thresholds(stats, strategy=strategy)
            assert t.min_avg_word_length <= t.max_avg_word_length


class TestArtifactRoundTrip:
    def test_write_and_load_stopwords(self, tmp_path):
        words = ["ve", "bir", "ışık"]
        t = TurkishQualityThresholds()
        write_artifacts(words, t, tmp_path)
        loaded = load_stopwords(tmp_path / STOPWORDS_FILENAME)
        assert loaded == frozenset(words)

    def test_write_and_load_thresholds_numeric_only(self, tmp_path):
        t = TurkishQualityThresholds(min_doc_words=42, max_avg_word_length=11.5)
        write_artifacts(["ve"], t, tmp_path, report={"n_docs": 3})
        loaded = load_thresholds(tmp_path / THRESHOLDS_FILENAME)
        assert loaded["min_doc_words"] == 42
        assert loaded["max_avg_word_length"] == 11.5
        # stop_words must NOT be persisted in the thresholds JSON.
        assert "stop_words" not in loaded

    def test_report_written_when_provided(self, tmp_path):
        write_artifacts(["ve"], TurkishQualityThresholds(), tmp_path, report={"n_docs": 7})
        report = load_thresholds(tmp_path / "calibration_report.json")
        assert report["n_docs"] == 7


class TestCalibrateFromCorpus:
    def _docs(self) -> list[str]:
        base = (
            "Türkiye Cumhuriyeti bir ülkedir ve başkenti Ankara şehridir. "
            "Bu yazıda evler ve kitaplar hakkında bilgi verilmektedir. "
            "Öğrenciler okula gittiler ve dersleri çalıştılar."
        )
        return [base + f" Bu {i}. paragraftır ve devam etmektedir." for i in range(20)]

    def test_end_to_end_returns_sensible_artifacts(self):
        stopwords, thresholds, report = calibrate_from_corpus(self._docs(), top_n=10)
        assert "ve" in stopwords
        assert len(stopwords) <= 10
        assert isinstance(thresholds, TurkishQualityThresholds)
        assert thresholds.min_avg_word_length <= thresholds.max_avg_word_length
        assert thresholds.stop_words == frozenset(stopwords)

    def test_report_provenance_fields(self):
        _, _, report = calibrate_from_corpus(self._docs(), top_n=10)
        assert report["n_docs"] == 20
        assert report["n_unique_words"] > 0
        assert "thresholds" in report
        assert "sample_quantiles" in report
        assert "stop_words" not in report["thresholds"]

    def test_empty_corpus_raises(self):
        # No documents -> cannot derive thresholds from nothing.
        with pytest.raises(ValueError, match="empty corpus"):
            calibrate_from_corpus([])

    def test_whitespace_only_corpus_raises(self):
        # All docs skipped by CorpusStats.add -> n_docs == 0 -> same guard fires.
        with pytest.raises(ValueError, match="empty corpus"):
            calibrate_from_corpus(["", "   ", "\n"])
