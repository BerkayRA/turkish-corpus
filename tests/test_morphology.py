"""Tests for the optional turkish-tokenizer (tr_api) bridge.

The tests that exercise the real morphological analyzer are skipped unless a local
turkish-tokenizer clone is found, so the suite stays green on a machine without it.
"""

import os

import pytest

from turkish_corpus.morphology import (
    TURKISH_TOKENIZER_PATH_ENV,
    MorphologicalSegmenter,
    ensure_tr_api_importable,
)

# Candidate local clones (dev convenience only — never required).
_CANDIDATES = [
    "/Users/berkayra/Downloads/tokenizer/turkish-tokenizer",
    os.path.join(os.path.expanduser("~"), "dev", "turkish-tokenizer"),
]


def _find_repo() -> str | None:
    env = os.environ.get(TURKISH_TOKENIZER_PATH_ENV)
    if env and os.path.isfile(os.path.join(env, "tr_api.py")):
        return env
    for path in _CANDIDATES:
        if os.path.isfile(os.path.join(path, "tr_api.py")):
            return path
    return None


class TestEnsureImportable:
    def test_raises_without_path_or_env(self, monkeypatch):
        monkeypatch.delenv(TURKISH_TOKENIZER_PATH_ENV, raising=False)
        with pytest.raises(FileNotFoundError, match="turkish-tokenizer"):
            ensure_tr_api_importable()

    def test_raises_for_dir_without_tr_api(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TURKISH_TOKENIZER_PATH_ENV, raising=False)
        with pytest.raises(FileNotFoundError, match="tr_api.py"):
            ensure_tr_api_importable(str(tmp_path))

    def test_uses_env_var(self, tmp_path, monkeypatch):
        (tmp_path / "tr_api.py").write_text("# stub\n")
        monkeypatch.setenv(TURKISH_TOKENIZER_PATH_ENV, str(tmp_path))
        # Should not raise; the dir gets onto sys.path.
        ensure_tr_api_importable()


@pytest.mark.skipif(_find_repo() is None, reason="turkish-tokenizer clone not available")
class TestSegmenterOnRealRepo:
    def test_segments_into_morphemes(self):
        segmenter = MorphologicalSegmenter(_find_repo())
        chunks = segmenter.segment("evlerimizden")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        # The root of "evlerimizden" is "ev"; it should appear among the morphemes.
        assert any("ev" in c for c in chunks)

    def test_count_matches_segment_length(self):
        segmenter = MorphologicalSegmenter(_find_repo())
        text = "kitaplarımı okudular"
        assert segmenter.count(text) == len(segmenter.segment(text))

    def test_empty_is_zero(self):
        segmenter = MorphologicalSegmenter(_find_repo())
        assert segmenter.count("") == 0
        assert segmenter.segment("") == []
