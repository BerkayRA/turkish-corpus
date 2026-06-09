"""Tests for the morpheme-aware BPE counter (turkish-llm's production tokenizer).

The pure :class:`MorphemeBPE` merge engine is exercised on synthetic merges (no ``tr_api``).
The :class:`MorphemeBPETokenCounter` and ``load_token_counter`` dispatch are exercised with a
*fake* ``tr_api`` module injected via monkeypatch, so they run without the optional
``turkish-tokenizer`` sibling repo (a real-repo smoke test is guarded by a skip).
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from turkish_corpus.morpheme_bpe import MorphemeBPE, MorphemeBPETokenCounter
from turkish_corpus.tokenizer import (
    HFTokenCounter,
    TokenCounter,
    load_token_counter,
)

# Where a real turkish-tokenizer clone might live for the optional smoke test.
_DEFAULT_REPO = os.path.expanduser("~/Downloads/tokenizer/turkish-tokenizer")
_REAL_REPO = os.environ.get("TURKISH_TOKENIZER_PATH") or _DEFAULT_REPO


def _install_fake_tr_api(monkeypatch, *, analyses=None):
    """Inject a fake ``tr_api`` module so the counter constructs without the real repo.

    ``analyses`` maps a word -> the dict its ``tokenize`` should return; unknown words get a
    default single-morpheme parse. Also neutralises ``ensure_tr_api_importable`` so no repo
    path resolution is attempted.
    """

    class _FakeTokenizer:
        def __init__(self, config):  # config object is irrelevant to the fake
            self.config = config

        def tokenize(self, word, **kwargs):
            if analyses is not None and word in analyses:
                return analyses[word]
            return {"parsed": True, "morphemes": [{"chunk": word}]}

    fake = types.ModuleType("tr_api")
    fake.Tokenizer = _FakeTokenizer
    fake.TokenizerConfig = lambda **kwargs: dict(kwargs)
    monkeypatch.setitem(sys.modules, "tr_api", fake)
    monkeypatch.setattr(
        "turkish_corpus.morphology.ensure_tr_api_importable",
        lambda repo_path=None: None,
    )


class TestMorphemeBPEEncode:
    def test_merges_to_single_token(self):
        bpe = MorphemeBPE([("a", "b"), ("ab", "c")])
        assert bpe.encode(["a", "b", "c"]) == ["abc"]

    def test_single_merge_leaves_neighbours(self):
        bpe = MorphemeBPE([("x", "y")])
        assert bpe.encode(["a", "x", "y", "z"]) == ["a", "xy", "z"]

    def test_lower_rank_wins_when_multiple_mergeable(self):
        # ("b","c") has rank 0 (higher priority) than ("a","b") at rank 1, so it merges first.
        bpe = MorphemeBPE([("b", "c"), ("a", "b")])
        assert bpe.encode(["a", "b", "c"]) == ["a", "bc"]

    def test_noop_when_nothing_merges(self):
        bpe = MorphemeBPE([("p", "q")])
        assert bpe.encode(["a", "b", "c"]) == ["a", "b", "c"]

    def test_single_morpheme_unchanged(self):
        assert MorphemeBPE([("a", "b")]).encode(["ev"]) == ["ev"]

    def test_empty_morpheme_list(self):
        assert MorphemeBPE([("a", "b")]).encode([]) == []


class TestFromFile:
    def test_round_trips_json(self, tmp_path):
        path = tmp_path / "morpheme_bpe_4.json"
        path.write_text(
            json.dumps({"morph_sep": "▁", "merges": [["a", "b"]]}),
            encoding="utf-8",
        )
        bpe = MorphemeBPE.from_file(str(path))
        assert bpe.ranks == {("a", "b"): 0}
        assert bpe.encode(["a", "b", "c"]) == ["ab", "c"]


class TestLoadTokenCounterDispatch:
    def test_morpheme_bpe_json_routes_to_morpheme_counter(self, tmp_path, monkeypatch):
        path = tmp_path / "morpheme_bpe_2.json"
        path.write_text(
            json.dumps({"morph_sep": "▁", "merges": [["ev", "ler"]]}),
            encoding="utf-8",
        )
        _install_fake_tr_api(monkeypatch)
        counter = load_token_counter(str(path))
        assert isinstance(counter, MorphemeBPETokenCounter)
        assert counter.name == str(path)

    def test_hf_shaped_json_routes_to_hf_counter(self, tmp_path):
        # HF tokenizer.json: top-level "model"/"version", NESTED merges, no "morph_sep".
        path = tmp_path / "tokenizer.json"
        path.write_text(
            json.dumps({"version": "1.0", "model": {"type": "BPE", "merges": ["a b"]}}),
            encoding="utf-8",
        )
        from unittest.mock import MagicMock, patch

        mock_tok = MagicMock()
        mock_tok.encode.return_value = MagicMock(ids=[1, 2])
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            counter = load_token_counter(str(path))
            assert isinstance(counter, HFTokenCounter)

    def test_unreadable_json_falls_back_to_hf(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{ not valid json", encoding="utf-8")
        from unittest.mock import MagicMock, patch

        mock_tok = MagicMock()
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            counter = load_token_counter(str(path))
            assert isinstance(counter, HFTokenCounter)


class TestMorphemeBPETokenCounter:
    def _counter(self, tmp_path, monkeypatch, *, merges, analyses=None):
        path = tmp_path / "morpheme_bpe.json"
        path.write_text(
            json.dumps({"morph_sep": "▁", "merges": merges}),
            encoding="utf-8",
        )
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        return MorphemeBPETokenCounter(str(path))

    def test_encode_word_uses_parsed_morphemes(self, tmp_path, monkeypatch):
        analyses = {
            "evler": {"parsed": True, "morphemes": [{"chunk": "ev"}, {"chunk": "ler"}]}
        }
        counter = self._counter(tmp_path, monkeypatch, merges=[["ev", "ler"]], analyses=analyses)
        assert counter.encode_word("evler") == ["evler"]

    def test_encode_word_falls_back_to_word_when_unparsed(self, tmp_path, monkeypatch):
        analyses = {"xyz": {"parsed": False}}
        counter = self._counter(tmp_path, monkeypatch, merges=[["a", "b"]], analyses=analyses)
        assert counter.encode_word("xyz") == ["xyz"]

    def test_count_sums_tokens_across_words(self, tmp_path, monkeypatch):
        analyses = {
            "evler": {"parsed": True, "morphemes": [{"chunk": "ev"}, {"chunk": "ler"}]},
            "gel": {"parsed": True, "morphemes": [{"chunk": "gel"}]},
        }
        # "ev"+"ler" merges to one token; "gel" is one token -> 2 total.
        counter = self._counter(tmp_path, monkeypatch, merges=[["ev", "ler"]], analyses=analyses)
        assert counter.count("evler gel") == 2

    def test_count_empty_is_zero(self, tmp_path, monkeypatch):
        counter = self._counter(tmp_path, monkeypatch, merges=[["a", "b"]])
        assert counter.count("") == 0

    def test_satisfies_token_counter_protocol(self, tmp_path, monkeypatch):
        counter = self._counter(tmp_path, monkeypatch, merges=[["a", "b"]])
        assert isinstance(counter, TokenCounter)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_REAL_REPO, "tr_api.py")),
    reason="turkish-tokenizer repo not locatable (set TURKISH_TOKENIZER_PATH)",
)
def test_real_tr_api_smoke(tmp_path):
    """Smoke test against the real ``tr_api`` when the sibling repo is present."""
    path = tmp_path / "morpheme_bpe.json"
    path.write_text(json.dumps({"morph_sep": "▁", "merges": []}), encoding="utf-8")
    counter = MorphemeBPETokenCounter(str(path), repo_path=_REAL_REPO)
    assert counter.count("evlerimizden gelenler") >= 2
