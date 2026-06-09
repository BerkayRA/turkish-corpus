"""Smoke tests for the datatrove pipeline assembly.

These require the optional ``pipeline`` extra and are skipped otherwise, so the core test
suite runs on a clean dev machine without the heavy install.
"""

import pytest

datatrove = pytest.importorskip("datatrove", reason="needs `uv sync --extra pipeline`")

pytestmark = pytest.mark.pipeline

import json  # noqa: E402

from turkish_corpus.config import default_hplt_config  # noqa: E402
from turkish_corpus.pipeline import (  # noqa: E402
    _lid_compatible,
    build_process_pipeline,
    build_token_counter_step,
)


def test_build_process_pipeline_returns_steps(tmp_path):
    cfg = default_hplt_config()
    cfg.reader.limit = 10
    cfg.output_path = str(tmp_path / "out")
    cfg.logging_dir = str(tmp_path / "logs")
    steps = build_process_pipeline(cfg)
    assert isinstance(steps, list)
    assert len(steps) >= 6
    names = " ".join(type(s).__name__ for s in steps)
    assert "Reader" in names
    assert "TurkishNormalizer" in names
    assert "TurkishPIIRedactor" in names


def test_language_filter_toggle(tmp_path):
    cfg = default_hplt_config()
    cfg.output_path = str(tmp_path / "out")
    cfg.language.enabled = False
    names = [type(s).__name__ for s in build_process_pipeline(cfg)]
    assert not any("LanguageFilter" in n for n in names)


def test_custom_blocks_import():
    from turkish_corpus.blocks import TurkishNormalizer, TurkishPIIRedactor

    assert TurkishNormalizer().type
    assert TurkishPIIRedactor().type


def test_language_filter_skipped_when_lid_incompatible(tmp_path, monkeypatch):
    # When fasttext's LID can't run (NumPy>=2), an enabled language filter degrades to a
    # warning + skip rather than crashing the pipeline at runtime.
    monkeypatch.setattr("turkish_corpus.pipeline._lid_compatible", lambda: False)
    cfg = default_hplt_config()
    cfg.output_path = str(tmp_path / "out")
    cfg.language.enabled = True
    names = [type(s).__name__ for s in build_process_pipeline(cfg)]
    assert not any("LanguageFilter" in n for n in names)


def test_language_filter_present_when_lid_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr("turkish_corpus.pipeline._lid_compatible", lambda: True)
    cfg = default_hplt_config()
    cfg.output_path = str(tmp_path / "out")
    cfg.language.enabled = True
    names = [type(s).__name__ for s in build_process_pipeline(cfg)]
    assert any("LanguageFilter" in n for n in names)


def test_lid_compatible_reflects_numpy_major():
    import numpy

    assert _lid_compatible() == (int(numpy.__version__.split(".")[0]) < 2)


def test_token_counter_step_none_when_unset():
    cfg = default_hplt_config()
    assert cfg.tokenizer.name_or_path is None
    assert build_token_counter_step(cfg) is None


def test_token_counter_step_morpheme_bpe_uses_turkish_counter(tmp_path):
    # morpheme_bpe-shaped json (top-level "morph_sep") -> TurkishTokensCounter.
    from turkish_corpus.blocks import TurkishTokensCounter

    spec = tmp_path / "morpheme_bpe_20000.json"
    spec.write_text(json.dumps({"morph_sep": "▁", "merges": []}), encoding="utf-8")
    cfg = default_hplt_config()
    cfg.tokenizer.name_or_path = str(spec)
    cfg.tokenizer.sample_rate = 0.5
    step = build_token_counter_step(cfg)
    assert isinstance(step, TurkishTokensCounter)
    assert step.sample_rate == 0.5
    assert step.tokenizer_spec == str(spec)


def test_token_counter_step_hf_json_uses_datatrove_counter(tmp_path):
    # HF-shaped tokenizer.json (nested merges, no "morph_sep") -> datatrove TokensCounter.
    from datatrove.pipeline.tokens import TokensCounter

    spec = tmp_path / "tokenizer.json"
    spec.write_text(
        json.dumps({"version": "1.0", "model": {"type": "BPE", "merges": ["a b"]}}),
        encoding="utf-8",
    )
    cfg = default_hplt_config()
    cfg.tokenizer.name_or_path = str(spec)
    step = build_token_counter_step(cfg)
    assert isinstance(step, TokensCounter)


def test_token_counter_step_sentencepiece_uses_turkish_counter():
    from turkish_corpus.blocks import TurkishTokensCounter

    cfg = default_hplt_config()
    cfg.tokenizer.name_or_path = "/models/sp_morph_32000.model"
    step = build_token_counter_step(cfg)
    assert isinstance(step, TurkishTokensCounter)


def test_token_counter_step_hub_id_uses_datatrove_counter():
    from datatrove.pipeline.tokens import TokensCounter

    cfg = default_hplt_config()
    cfg.tokenizer.name_or_path = "dbmdz/bert-base-turkish-cased"
    step = build_token_counter_step(cfg)
    assert isinstance(step, TokensCounter)
