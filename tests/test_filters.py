"""Tests for the Turkish quality-filter parameters and their derived-artifact loading.

A calibration artifact (derived from Turkish Wikipedia) IS committed under
``turkish_corpus/data/``, so the loaders resolve to the derived values; the fallback to the
curated list / dataclass defaults is exercised by monkeypatching the artifact away. The
agglutination-widened mean-word-length band is pinned so a refactor cannot quietly narrow it
back to English, and ``max_doc_words`` must stay generous so long-form theses aren't dropped.
"""

import pytest

from turkish_corpus import filters as filters_mod
from turkish_corpus.filters import TURKISH_STOPWORDS, TurkishQualityThresholds


class _MissingResource:
    """A resource handle whose read_text always raises, to simulate an absent artifact."""

    def read_text(self, *a, **k):
        raise FileNotFoundError("simulated missing artifact")


class _FileResource:
    """A resource handle that reads a real tmp file, to simulate a committed (bad) artifact."""

    def __init__(self, path):
        self._path = path

    def read_text(self, *a, **k):
        return self._path.read_text(encoding="utf-8")


class TestStopwords:
    def test_non_empty(self):
        assert len(TURKISH_STOPWORDS) > 0

    def test_contains_core_function_words(self):
        # The derived list (and the curated fallback) lead with the most frequent Turkish
        # function words.
        assert "ve" in TURKISH_STOPWORDS
        assert "bir" in TURKISH_STOPWORDS

    def test_is_frozenset(self):
        assert isinstance(TURKISH_STOPWORDS, frozenset)


class TestThresholdDefaults:
    def test_mean_word_length_widened_for_agglutination(self):
        # The whole point of the Turkish tuning: upper band > English's 10.0.
        assert TurkishQualityThresholds().max_avg_word_length > 10.0

    def test_calibrated_loads_committed_artifact(self):
        # The committed Wikipedia-derived artifact must actually be loaded — its values
        # differ from the dataclass defaults (50 / 12.0).
        calibrated = TurkishQualityThresholds.calibrated()
        default = TurkishQualityThresholds()
        assert calibrated.min_doc_words != default.min_doc_words
        assert calibrated.max_avg_word_length < default.max_avg_word_length  # ~8.18 < 12.0

    def test_max_doc_words_stays_generous_for_long_form(self):
        # Derived from short Wikipedia articles, but must NOT truncate long-form theses/laws.
        assert TurkishQualityThresholds.calibrated().max_doc_words >= 100_000

    def test_calibrated_falls_back_to_defaults_when_absent(self, monkeypatch):
        # Simulate a fresh checkout before calibration: loader can't find the artifact.
        monkeypatch.setattr(filters_mod, "_data_resource", lambda _f: _MissingResource())
        calibrated = TurkishQualityThresholds.calibrated()
        default = TurkishQualityThresholds()
        assert calibrated.min_doc_words == default.min_doc_words
        assert calibrated.max_avg_word_length == default.max_avg_word_length

    def test_calibrated_uses_module_stopwords(self):
        assert TurkishQualityThresholds.calibrated().stop_words == TURKISH_STOPWORDS

    def test_calibrated_falls_back_on_invalid_json(self, tmp_path, monkeypatch):
        # A corrupt/unparseable committed artifact must fail SAFE to defaults, not crash at
        # import time (config defaults to calibrated()).
        bad = tmp_path / "turkish_thresholds.json"
        bad.write_text("{ this is not valid json", encoding="utf-8")
        monkeypatch.setattr(filters_mod, "_data_resource", lambda _f: _FileResource(bad))
        calibrated = TurkishQualityThresholds.calibrated()
        default = TurkishQualityThresholds()
        assert calibrated.min_doc_words == default.min_doc_words
        assert calibrated.max_avg_word_length == default.max_avg_word_length

    def test_calibrated_drops_wrong_typed_field(self, tmp_path, monkeypatch):
        # A wrong-typed field (string where a number is expected) must be dropped, never
        # reaching __post_init__; the rest of the object stays at defaults.
        bad = tmp_path / "turkish_thresholds.json"
        bad.write_text('{"min_doc_words": "bad"}', encoding="utf-8")
        monkeypatch.setattr(filters_mod, "_data_resource", lambda _f: _FileResource(bad))
        calibrated = TurkishQualityThresholds.calibrated()
        assert calibrated.min_doc_words == TurkishQualityThresholds().min_doc_words


class TestGopherKwargs:
    def test_shape_intact(self):
        kwargs = TurkishQualityThresholds().as_gopher_kwargs()
        expected = {
            "min_doc_words", "max_doc_words", "min_avg_word_length", "max_avg_word_length",
            "max_symbol_word_ratio", "max_bullet_lines_ratio", "max_ellipsis_lines_ratio",
            "max_non_alpha_words_ratio", "min_stop_words", "stop_words",
        }
        assert set(kwargs) == expected

    def test_stop_words_serialized_as_list(self):
        kwargs = TurkishQualityThresholds().as_gopher_kwargs()
        assert isinstance(kwargs["stop_words"], list)


class TestValidation:
    def test_rejects_inverted_word_count_band(self):
        with pytest.raises(ValueError, match="min_doc_words"):
            TurkishQualityThresholds(min_doc_words=100, max_doc_words=10)

    def test_rejects_inverted_word_length_band(self):
        with pytest.raises(ValueError, match="min_avg_word_length"):
            TurkishQualityThresholds(min_avg_word_length=15.0, max_avg_word_length=5.0)

    def test_rejects_out_of_range_ratio(self):
        with pytest.raises(ValueError, match="max_symbol_word_ratio"):
            TurkishQualityThresholds(max_symbol_word_ratio=1.5)
