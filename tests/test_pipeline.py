"""Smoke tests for the datatrove pipeline assembly.

These require the optional ``pipeline`` extra and are skipped otherwise, so the core test
suite runs on a clean dev machine without the heavy install.
"""

import pytest

datatrove = pytest.importorskip("datatrove", reason="needs `uv sync --extra pipeline`")

pytestmark = pytest.mark.pipeline

from turkish_corpus.config import default_hplt_config  # noqa: E402
from turkish_corpus.pipeline import (  # noqa: E402
    _lid_compatible,
    build_process_pipeline,
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
