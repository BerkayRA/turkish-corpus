"""Tests for the hardened morpheme segmenter (``turkish_corpus.morph_segment``).

The pure core — :func:`segment_word` (incl. its SIGALRM hang-proofing), :func:`segment_line`,
:func:`build_cache`, and the dedup+reconstruct pipeline — is exercised entirely in-process with
*fake* tokenizers and an injectable per-word function: no real ``tr_api`` and no real
multiprocessing are needed. A real-``tr_api`` smoke test is guarded by a skip, and the
``run_segmentation`` end-to-end runs single-worker against a monkeypatched ``tr_api`` module
(the same pattern ``tests/test_morpheme_bpe.py`` uses).
"""

from __future__ import annotations

import os
import signal
import sys
import time
import types

import pytest

from turkish_corpus.morph_segment import (
    MORPH_SEP,
    build_cache,
    install_alarm_handler,
    run_segmentation,
    segment_corpus,
    segment_line,
    segment_word,
)

# Where a real turkish-tokenizer clone might live for the optional smoke test.
_DEFAULT_REPO = os.path.expanduser("~/Downloads/tokenizer/turkish-tokenizer")
_REAL_REPO = os.environ.get("TURKISH_TOKENIZER_PATH") or _DEFAULT_REPO


class _FakeTokenizer:
    """Records calls and returns canned analyses keyed by word.

    ``analyses`` maps word -> the dict ``tokenize`` should return; unknown words get a default
    single-morpheme parse. ``calls`` is the list of words actually analysed (so tests can
    assert tokenize was / was not called).
    """

    def __init__(self, analyses=None):
        self.analyses = analyses or {}
        self.calls: list[str] = []

    def tokenize(self, word, **kwargs):
        self.calls.append(word)
        if word in self.analyses:
            return self.analyses[word]
        return {"parsed": True, "morphemes": [{"chunk": word}]}


class TestSegmentWord:
    def test_joins_parsed_morphemes_with_separator(self):
        tok = _FakeTokenizer(
            {"geliyorum": {"parsed": True, "morphemes": [
                {"chunk": "gel"}, {"chunk": "iyor"}, {"chunk": "um"}]}}
        )
        assert segment_word("geliyorum", tok) == f"gel{MORPH_SEP}iyor{MORPH_SEP}um"

    def test_unparsed_word_passes_through_as_single_token(self):
        tok = _FakeTokenizer({"xyzq": {"parsed": False}})
        assert segment_word("xyzq", tok) == "xyzq"

    def test_empty_morphemes_falls_back_to_word(self):
        tok = _FakeTokenizer({"ev": {"parsed": True, "morphemes": []}})
        assert segment_word("ev", tok) == "ev"

    def test_long_word_returned_unchanged_without_calling_tokenizer(self):
        tok = _FakeTokenizer()
        long_word = "a" * 80  # over default max_len (70)
        assert segment_word(long_word, tok, max_len=70) == long_word
        assert tok.calls == []  # the chart parser was never invoked on the junk token

    def test_word_at_max_len_is_still_analysed(self):
        tok = _FakeTokenizer()
        word = "a" * 70
        assert segment_word(word, tok, max_len=70) == word
        assert tok.calls == [word]  # len == max_len is NOT over the cap


class TestSegmentWordHardening:
    """The four-requirement hang-proofing, validated in-process via real SIGALRM."""

    def test_timeout_returns_passthrough_well_under_sleep(self):
        install_alarm_handler()

        class _SlowTok:
            def tokenize(self, word, **kw):
                time.sleep(3)  # would hang the batch without the per-word alarm
                return {"parsed": True, "morphemes": [{"chunk": word}]}

        start = time.monotonic()
        result = segment_word("stall", _SlowTok(), timeout=0.3)
        elapsed = time.monotonic() - start

        assert result == "stall"  # passthrough, not a crash or a hang
        assert elapsed < 1.0  # abandoned ~0.3s in, nowhere near the 3s sleep

    def test_tokenizer_exception_returns_passthrough(self):
        class _BoomTok:
            def tokenize(self, word, **kw):
                raise ValueError("chart parser exploded")

        assert segment_word("boom", _BoomTok()) == "boom"

    def test_timer_disarmed_after_call(self):
        # A fast word leaves no pending timer; a subsequent slow-but-untimed op must not fire.
        tok = _FakeTokenizer()
        assert segment_word("ev", tok) == "ev"
        # If the timer leaked, this sleep would raise _SegTimeout; it must complete cleanly.
        time.sleep(0.05)

    def test_handler_is_noop_when_disarmed(self):
        # Regression: the SIGALRM that fired during the finally's setitimer(0) disarm used to
        # escape uncaught and kill the whole multiprocessing pool. The armed-flag guard makes a
        # stray/late delivery a no-op. After a normal call the flag is cleared, so invoking the
        # handler directly (simulating that late delivery) must NOT raise.
        import turkish_corpus.morph_segment as ms

        segment_word("ev", _FakeTokenizer())
        assert ms._alarm_armed is False
        ms._on_alarm(signal.SIGALRM, None)  # must return cleanly, not raise _SegTimeout

    def test_handler_raises_only_while_armed(self):
        import turkish_corpus.morph_segment as ms

        ms._alarm_armed = True
        try:
            with pytest.raises(ms._SegTimeout):
                ms._on_alarm(signal.SIGALRM, None)
        finally:
            ms._alarm_armed = False


class TestSegmentLine:
    def test_reconstructs_from_cache_joined_with_spaces(self):
        cache = {"gel": f"gel{MORPH_SEP}di", "ev": f"ev{MORPH_SEP}de"}
        assert segment_line("gel ev", cache) == f"gel{MORPH_SEP}di ev{MORPH_SEP}de"

    def test_blank_line_returns_none(self):
        assert segment_line("   ", {}) is None

    def test_empty_line_returns_none(self):
        assert segment_line("", {}) is None

    def test_missing_word_falls_back_to_itself(self):
        assert segment_line("unknown", {}) == "unknown"


class TestBuildCache:
    def test_segments_each_unique_word_once(self):
        seen: list[str] = []

        def fake_seg(word: str) -> str:
            seen.append(word)
            return word.upper()

        cache = build_cache({"a", "b"}, fake_seg)
        assert cache == {"a": "A", "b": "B"}
        assert sorted(seen) == ["a", "b"]


class TestSegmentCorpusDedup:
    """End-to-end dedup+reconstruct with an injected per-word segmenter (no tr_api / no pool)."""

    def test_one_line_in_one_line_out_with_format(self):
        def seg(word: str) -> str:
            return {"geliyorum": f"gel{MORPH_SEP}iyor{MORPH_SEP}um"}.get(word, word)

        out, _ = segment_corpus(["geliyorum kitabı"], seg)
        lines = list(out)
        assert lines == [f"gel{MORPH_SEP}iyor{MORPH_SEP}um kitabı"]

    def test_repeated_word_segmented_once(self):
        calls: list[str] = []

        def seg(word: str) -> str:
            calls.append(word)
            return word + "X"

        out, cache = segment_corpus(["ev ev ev", "ev"], seg)
        list(out)  # drain
        assert calls.count("ev") == 1  # unique form analysed exactly once
        assert cache["ev"] == "evX"

    def test_long_word_passed_through(self):
        long_word = "h" * 90

        # segment_corpus does not itself enforce max_len on the segment_fn, but the unique-word
        # collection keeps the long word; the injected fn mimics segment_word's passthrough.
        def seg(word: str) -> str:
            return word if len(word) > 70 else word + "+"

        out, cache = segment_corpus([f"{long_word} ok"], seg)
        assert list(out) == [f"{long_word} ok+"]
        assert cache[long_word] == long_word

    def test_blank_lines_dropped(self):
        out, _ = segment_corpus(["", "  ", "ev"], lambda w: w)
        assert list(out) == ["ev"]


def _install_fake_tr_api(monkeypatch, *, analyses=None):
    """Inject a fake ``tr_api`` module so ``run_segmentation`` builds without the real repo."""

    class _FakeTok:
        def __init__(self, config):
            self.config = config

        def tokenize(self, word, **kwargs):
            if analyses and word in analyses:
                return analyses[word]
            return {"parsed": True, "morphemes": [{"chunk": word}]}

    fake = types.ModuleType("tr_api")
    fake.Tokenizer = _FakeTok
    fake.TokenizerConfig = lambda **kwargs: dict(kwargs)
    monkeypatch.setitem(sys.modules, "tr_api", fake)
    monkeypatch.setattr(
        "turkish_corpus.morphology.ensure_tr_api_importable",
        lambda repo_path=None: None,
    )


class TestRunSegmentationSingleWorker:
    """``run_segmentation`` with workers=1 (in-process) against a monkeypatched ``tr_api``."""

    def test_end_to_end_format_and_stats(self, tmp_path, monkeypatch):
        long_word = "z" * 90
        analyses = {
            "geliyorum": {"parsed": True, "morphemes": [
                {"chunk": "gel"}, {"chunk": "iyor"}, {"chunk": "um"}]},
            "kitabı": {"parsed": True, "morphemes": [{"chunk": "kitab"}, {"chunk": "ı"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)

        src = tmp_path / "corpus.txt"
        # 'geliyorum' repeats (cache hit), a blank line is dropped, long_word passes through.
        src.write_text(
            f"geliyorum kitabı\n\ngeliyorum {long_word}\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.morph.txt"

        stats = run_segmentation(src, out, workers=1, repo_path=None)

        produced = out.read_text(encoding="utf-8").splitlines()
        assert produced == [
            f"gel{MORPH_SEP}iyor{MORPH_SEP}um kitab{MORPH_SEP}ı",
            f"gel{MORPH_SEP}iyor{MORPH_SEP}um {long_word}",
        ]
        assert stats["lines_in"] == 3
        assert stats["lines_out"] == 2
        assert stats["lines_dropped"] == 1
        # unique words: geliyorum, kitabı, long_word
        assert stats["unique_words"] == 3
        # long_word is the only passthrough (no separator, equals itself)
        assert stats["passthrough_words"] == 1

    def test_max_lines_caps_input(self, tmp_path, monkeypatch):
        _install_fake_tr_api(monkeypatch)
        src = tmp_path / "corpus.txt"
        src.write_text("ev\nev\nev\n", encoding="utf-8")
        out = tmp_path / "out.txt"
        stats = run_segmentation(src, out, workers=1, max_lines=2)
        assert stats["lines_in"] == 2


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_REAL_REPO, "tr_api.py")),
    reason="turkish-tokenizer repo not locatable (set TURKISH_TOKENIZER_PATH)",
)
def test_real_tr_api_smoke(tmp_path):
    """Smoke test against the real ``tr_api`` when the sibling repo is present (single worker)."""
    src = tmp_path / "corpus.txt"
    src.write_text("evlerimizden gelenler\n", encoding="utf-8")
    out = tmp_path / "out.morph.txt"
    stats = run_segmentation(src, out, workers=1, repo_path=_REAL_REPO)
    assert stats["lines_out"] == 1
    produced = out.read_text(encoding="utf-8").strip()
    assert " " in produced  # two words preserved as space-separated
