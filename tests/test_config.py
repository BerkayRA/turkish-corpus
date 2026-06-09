"""Tests for pipeline configuration and validation."""

import pytest

from turkish_corpus.config import HPLT_DATASET, HPLT_LANG, default_hplt_config
from turkish_corpus.filters import TurkishQualityThresholds


class TestDefaults:
    def test_defaults_validate(self):
        cfg = default_hplt_config()
        assert cfg.reader.dataset == HPLT_DATASET
        assert cfg.reader.language == HPLT_LANG
        assert cfg.reader.source == "hf"

    def test_minhash_finewebt2_settings(self):
        cfg = default_hplt_config()
        assert cfg.minhash.num_buckets == 14
        assert cfg.minhash.hashes_per_bucket == 8
        assert cfg.minhash.n_grams == 5

    def test_turkish_avg_word_length_widened(self):
        # Agglutination: the upper bound must exceed the English default of 10.
        assert TurkishQualityThresholds().max_avg_word_length > 10.0


class TestValidation:
    def test_jsonl_requires_data_path(self):
        cfg = default_hplt_config()
        cfg.reader.source = "jsonl"
        cfg.reader.data_path = None
        with pytest.raises(ValueError, match="data_path"):
            cfg.validate()

    def test_bad_source_rejected(self):
        cfg = default_hplt_config()
        cfg.reader.source = "ftp"
        with pytest.raises(ValueError, match="reader.source"):
            cfg.validate()

    def test_threshold_range(self):
        cfg = default_hplt_config()
        cfg.language.threshold = 1.5
        with pytest.raises(ValueError, match="threshold"):
            cfg.validate()

    def test_unknown_override_rejected(self):
        with pytest.raises(AttributeError):
            default_hplt_config(nonexistent_field=123)

    def test_minhash_bounds_checked(self):
        cfg = default_hplt_config()
        cfg.minhash.num_buckets = 0
        with pytest.raises(ValueError, match="minhash"):
            cfg.validate()

    def test_sample_rate_zero_rejected(self):
        cfg = default_hplt_config()
        cfg.tokenizer.sample_rate = 0
        with pytest.raises(ValueError, match="sample_rate"):
            cfg.validate()

    def test_sample_rate_above_one_rejected(self):
        cfg = default_hplt_config()
        cfg.tokenizer.sample_rate = 1.5
        with pytest.raises(ValueError, match="sample_rate"):
            cfg.validate()
