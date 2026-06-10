"""Tests for the production morpheme-BPE tokenizer (:class:`MorphemeTokenizer`).

The vocab/encode/decode/table/save-load surface is exercised with synthetic merges, a small
vocabulary, and a *fake* ``tr_api`` module injected via monkeypatch (mirroring
``tests/test_morpheme_bpe.py``), so nothing needs the real ``turkish-tokenizer`` repo or any
network. A fake tokenizer that counts ``tokenize`` calls proves the fast-table / cache paths
never touch the analyzer.
"""

from __future__ import annotations

import collections
import json
import sys
import types

import pytest

from turkish_corpus.morpheme_bpe import MorphemeBPE
from turkish_corpus.morpheme_tokenizer import (
    DEFAULT_VOCAB_SIZE,
    MORPH_SEP,
    MorphemeTokenizer,
)

# ---------------------------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------------------------


def _install_fake_tr_api(monkeypatch, *, analyses=None, calls=None):
    """Inject a fake ``tr_api`` so the analyzer path runs without the real repo.

    ``analyses`` maps word -> the dict ``tokenize`` returns (default: single-morpheme parse).
    ``calls`` (optional dict with key ``"n"``) is incremented on every ``tokenize`` so a test
    can assert the analyzer was / was not invoked. Also neutralises
    ``ensure_tr_api_importable`` so no repo-path resolution is attempted.
    """

    class _FakeTokenizer:
        def __init__(self, config):  # config is irrelevant to the fake
            self.config = config

        def tokenize(self, word, **kwargs):
            if calls is not None:
                calls["n"] += 1
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


def _merges_file(tmp_path, merges):
    """Write a ``morpheme_bpe.json`` (top-level ``morph_sep`` + ordered ``merges``)."""
    path = tmp_path / "morpheme_bpe.json"
    path.write_text(
        json.dumps({"morph_sep": MORPH_SEP, "merges": merges}),
        encoding="utf-8",
    )
    return str(path)


# A tiny vocab: pieces "ev", "gel", "di", "ler" with synthetic frequencies.
_PIECE_FREQS = collections.Counter({"ev": 100, "gel": 80, "di": 60, "ler": 40})
_SMALL_VOCAB = 261 + 4  # specials(5) + bytes(256) + 4 morpheme pieces


def _build(tmp_path, *, merges=None, piece_freqs=None, vocab_size=_SMALL_VOCAB, table=None):
    """Build a MorphemeTokenizer from a synthetic merges file + piece Counter."""
    return MorphemeTokenizer.build(
        merges_path=_merges_file(tmp_path, merges or []),
        piece_freqs=piece_freqs if piece_freqs is not None else _PIECE_FREQS,
        vocab_size=vocab_size,
        table=table,
    )


# ---------------------------------------------------------------------------------------------
# Vocab layout
# ---------------------------------------------------------------------------------------------


class TestVocabLayout:
    def test_specials_at_0_to_4(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.piece_to_id[("<special>", "PAD")] == 0
        assert tok.piece_to_id[("<special>", "UNK")] == 1
        assert tok.piece_to_id[("<special>", "BOS")] == 2
        assert tok.piece_to_id[("<special>", "EOS")] == 3
        assert tok.piece_to_id[("<special>", "WORD_BOUNDARY")] == 4

    def test_256_byte_tokens_at_5_to_260(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.piece_to_id[("<byte>", 0)] == 5
        assert tok.piece_to_id[("<byte>", 255)] == 260
        # All 256 byte ids are contiguous in [5, 260].
        byte_ids = [tok.piece_to_id[("<byte>", v)] for v in range(256)]
        assert byte_ids == list(range(5, 261))

    def test_pieces_start_at_261_in_frequency_order(self, tmp_path):
        tok = _build(tmp_path)
        # Most frequent first: ev(100) gel(80) di(60) ler(40).
        assert tok.id_to_piece[261] == "ev"
        assert tok.id_to_piece[262] == "gel"
        assert tok.id_to_piece[263] == "di"
        assert tok.id_to_piece[264] == "ler"

    def test_piece_to_id_and_id_to_piece_consistent(self, tmp_path):
        tok = _build(tmp_path)
        for tid, piece in enumerate(tok.id_to_piece):
            assert tok.piece_to_id[piece] == tid

    def test_vocab_size_matches_layout(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.vocab_size == _SMALL_VOCAB


# ---------------------------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------------------------


class TestEncode:
    def test_word_boundary_prepended_per_word(self, tmp_path, monkeypatch):
        _install_fake_tr_api(monkeypatch)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("ev gel")
        # Two words -> two WORD_BOUNDARY markers, each before its piece.
        assert ids[0] == wb
        assert ids.count(wb) == 2

    def test_known_pieces_map_to_their_ids(self, tmp_path, monkeypatch):
        analyses = {
            "evler": {"parsed": True, "morphemes": [{"chunk": "ev"}, {"chunk": "ler"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("evler")
        assert ids == [wb, tok.piece_to_id["ev"], tok.piece_to_id["ler"]]

    def test_oov_morpheme_falls_back_to_utf8_byte_ids(self, tmp_path, monkeypatch):
        # "zzz" is not in the vocab -> its UTF-8 bytes become byte-token ids.
        analyses = {"zzz": {"parsed": True, "morphemes": [{"chunk": "zzz"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        byte_off = tok.piece_to_id[("<byte>", 0)]
        ids = tok.encode("zzz")
        expected = [wb] + [byte_off + b for b in b"zzz"]
        assert ids == expected

    def test_long_word_uses_byte_fallback(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        byte_off = tok.piece_to_id[("<byte>", 0)]
        long_word = "a" * 80  # > MAX_WORD_LEN (70)
        ids = tok.encode(long_word)
        expected = [wb] + [byte_off + b for b in long_word.encode("utf-8")]
        assert ids == expected
        assert calls["n"] == 0  # never sent to the analyzer

    def test_add_bos_eos(self, tmp_path, monkeypatch):
        _install_fake_tr_api(monkeypatch)
        tok = _build(tmp_path)
        bos = tok.piece_to_id[("<special>", "BOS")]
        eos = tok.piece_to_id[("<special>", "EOS")]
        ids = tok.encode("ev", add_bos=True, add_eos=True)
        assert ids[0] == bos and ids[-1] == eos


# ---------------------------------------------------------------------------------------------
# Round-trip (decode . encode)
# ---------------------------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trips_known_pieces(self, tmp_path, monkeypatch):
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "gel": {"parsed": True, "morphemes": [{"chunk": "gel"}]},
            "di": {"parsed": True, "morphemes": [{"chunk": "di"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        assert tok.decode(tok.encode("ev gel di")) == "ev gel di"

    def test_round_trips_oov_via_byte_fallback(self, tmp_path, monkeypatch):
        # "ev" is known; "zzz" is OOV and round-trips through byte-fallback.
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "zzz": {"parsed": True, "morphemes": [{"chunk": "zzz"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        assert tok.decode(tok.encode("ev zzz")) == "ev zzz"

    def test_round_trips_multibyte_turkish_word(self, tmp_path, monkeypatch):
        # "ışığı" is OOV here -> multi-byte UTF-8 bytes split across byte tokens, reassembled.
        word = "ışığı"
        analyses = {word: {"parsed": True, "morphemes": [{"chunk": word}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        assert tok.decode(tok.encode(word)) == word

    def test_round_trips_mixed_known_and_multibyte_oov(self, tmp_path, monkeypatch):
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "ışık": {"parsed": True, "morphemes": [{"chunk": "ışık"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        assert tok.decode(tok.encode("ev ışık")) == "ev ışık"

    def test_decode_drops_bos_eos_pad(self, tmp_path, monkeypatch):
        _install_fake_tr_api(monkeypatch)
        tok = _build(tmp_path)
        ids = tok.encode("ev", add_bos=True, add_eos=True)
        pad = tok.piece_to_id[("<special>", "PAD")]
        assert tok.decode([pad] + ids + [pad]) == "ev"


# ---------------------------------------------------------------------------------------------
# Fast table
# ---------------------------------------------------------------------------------------------


class TestFastTable:
    def test_tabled_word_skips_analyzer(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("evler")
        assert ids == [wb, tok.piece_to_id["ev"], tok.piece_to_id["ler"]]
        assert calls["n"] == 0  # the table satisfied the lookup, analyzer untouched

    def test_non_tabled_word_falls_back_to_analyzer(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        analyses = {"gel": {"parsed": True, "morphemes": [{"chunk": "gel"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses, calls=calls)
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        tok.encode("gel")  # not in the table
        assert calls["n"] == 1

    def test_repeated_non_tabled_word_analyzed_once(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = _build(tmp_path)
        tok.encode("ev ev ev ev")
        assert calls["n"] == 1  # cached after the first analysis
        info = tok.cache_info()
        assert info.hits == 3 and info.misses == 1

    def test_cache_disabled_analyzes_each_time(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = MorphemeTokenizer.build(
            merges_path=_merges_file(tmp_path, []),
            piece_freqs=_PIECE_FREQS,
            vocab_size=_SMALL_VOCAB,
            cache_size=0,
        )
        tok.encode("ev ev ev")
        assert calls["n"] == 3
        assert tok.cache_info() is None


# ---------------------------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trips_vocab_and_merges(self, tmp_path):
        tok = _build(tmp_path, merges=[["ev", "ler"]])
        out = tmp_path / "saved"
        tok.save(str(out))
        loaded = MorphemeTokenizer.load(str(out))
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.id_to_piece == tok.id_to_piece
        assert loaded.piece_to_id == tok.piece_to_id
        # The merge engine is rebuilt and behaves identically.
        assert loaded._bpe.encode(["ev", "ler"]) == MorphemeBPE([("ev", "ler")]).encode(
            ["ev", "ler"]
        )

    def test_round_trips_table(self, tmp_path):
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        out = tmp_path / "saved"
        tok.save(str(out))
        assert (out / "table.tsv").is_file()
        loaded = MorphemeTokenizer.load(str(out))
        assert loaded.has_table
        assert loaded._table["evler"] == ["ev", "ler"]

    def test_no_table_file_when_no_table(self, tmp_path):
        tok = _build(tmp_path)
        out = tmp_path / "saved"
        tok.save(str(out))
        assert not (out / "table.tsv").exists()

    def test_load_does_not_require_tr_api_until_needed(self, tmp_path, monkeypatch):
        # Save a tokenizer with a table covering the word we will encode.
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        out = tmp_path / "saved"
        tok.save(str(out))

        # Make ensure_tr_api_importable explode if it is ever called.
        def _boom(repo_path=None):
            raise AssertionError("tr_api should not be needed for tabled/saved-load path")

        monkeypatch.setattr("turkish_corpus.morphology.ensure_tr_api_importable", _boom)

        loaded = MorphemeTokenizer.load(str(out))  # must not touch tr_api
        wb = loaded.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = loaded.encode("evler")  # tabled -> still no tr_api
        assert ids == [wb, loaded.piece_to_id["ev"], loaded.piece_to_id["ler"]]
        assert loaded.decode(ids) == "evler"

    def test_loaded_tokenizer_round_trips_text(self, tmp_path):
        tok = _build(tmp_path, table={"ev": ["ev"], "gel": ["gel"], "di": ["di"]})
        out = tmp_path / "saved"
        tok.save(str(out))
        loaded = MorphemeTokenizer.load(str(out))
        assert loaded.decode(loaded.encode("ev gel di")) == "ev gel di"


# ---------------------------------------------------------------------------------------------
# build() factory
# ---------------------------------------------------------------------------------------------


class TestBuildFactory:
    def test_respects_vocab_size(self, tmp_path):
        # Room for only 2 morpheme pieces beyond specials + bytes.
        vocab_size = 261 + 2
        tok = _build(tmp_path, vocab_size=vocab_size)
        assert tok.vocab_size == vocab_size
        # Only the two most frequent pieces made the cut.
        assert tok.id_to_piece[261] == "ev"
        assert tok.id_to_piece[262] == "gel"
        assert "di" not in tok.piece_to_id
        assert "ler" not in tok.piece_to_id

    def test_frequency_ordering(self, tmp_path):
        freqs = collections.Counter({"rare": 1, "common": 1000, "mid": 50})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=261 + 3)
        assert tok.id_to_piece[261:264] == ["common", "mid", "rare"]

    def test_tie_broken_by_piece_string(self, tmp_path):
        freqs = collections.Counter({"b": 10, "a": 10, "c": 10})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=261 + 3)
        # Equal frequency -> alphabetical, for reproducibility.
        assert tok.id_to_piece[261:264] == ["a", "b", "c"]

    def test_rejects_too_small_vocab(self, tmp_path):
        with pytest.raises(ValueError, match="too small"):
            _build(tmp_path, vocab_size=261)  # no room for any morpheme pieces

    def test_default_vocab_size_is_64000(self):
        assert DEFAULT_VOCAB_SIZE == 64_000
