"""Tests for the production morpheme-BPE tokenizer (:class:`MorphemeTokenizer`).

The vocab/encode/decode/table/save-load surface is exercised with synthetic morpheme merges,
synthetic byte-BPE merges, a small vocabulary, and a *fake* ``tr_api`` module injected via
monkeypatch (mirroring ``tests/test_morpheme_bpe.py``), so nothing needs the real
``turkish-tokenizer`` repo or any network. A fake tokenizer that counts ``tokenize`` calls
proves the fast-table / cache paths never touch the analyzer.

The headline guarantee under test is **lossless round-trip**: ``decode(encode(text)) == text``
for lowercase, Titlecase, ALLCAPS, apostrophe'd, mixed-case, foreign/emoji, and multi-word
inputs — including a fake tr_api that drops an apostrophe so the fidelity-check -> byte-BPE
path is exercised.
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
    bytes_to_unicode,
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


# A tiny morpheme vocab under the FUSED word-boundary scheme: word-initial forms are
# ``▁``-prefixed ("▁ev", "▁gel"), word-internal forms are bare ("di", "ler").
_PIECE_FREQS = collections.Counter(
    {
        MORPH_SEP + "ev": 100,
        MORPH_SEP + "gel": 80,
        "di": 60,
        "ler": 40,
    }
)

# Layout ids: 7 specials + 256 byte-chars => first byte-BPE id is 263. With no byte merges,
# the morpheme pieces start at 263. The default fixtures use NO byte merges unless asked.
_FIRST_MORPH_ID = 7 + 256  # == 263
_SMALL_VOCAB = _FIRST_MORPH_ID + 4  # + 4 morpheme pieces


def _build(
    tmp_path,
    *,
    merges=None,
    piece_freqs=None,
    byte_merges=None,
    vocab_size=_SMALL_VOCAB,
    table=None,
):
    """Build a MorphemeTokenizer from synthetic merges + a piece Counter + byte merges."""
    return MorphemeTokenizer.build(
        merges_path=_merges_file(tmp_path, merges or []),
        piece_freqs=piece_freqs if piece_freqs is not None else _PIECE_FREQS,
        byte_merges=byte_merges if byte_merges is not None else [],
        vocab_size=vocab_size,
        table=table,
    )


# ---------------------------------------------------------------------------------------------
# bytes_to_unicode bijection
# ---------------------------------------------------------------------------------------------


class TestByteUnicodeBijection:
    def test_covers_all_256_bytes(self):
        mapping = bytes_to_unicode()
        assert len(mapping) == 256
        assert sorted(mapping.keys()) == list(range(256))

    def test_is_injective(self):
        mapping = bytes_to_unicode()
        assert len(set(mapping.values())) == 256

    def test_no_whitespace_or_control_chars(self):
        # byte-chars must be safe to store in a TSV / split on whitespace.
        for char in bytes_to_unicode().values():
            assert not char.isspace()
            assert char.isprintable()


# ---------------------------------------------------------------------------------------------
# Vocab layout
# ---------------------------------------------------------------------------------------------


class TestVocabLayout:
    def test_specials_at_0_to_6(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.piece_to_id[("<special>", "PAD")] == 0
        assert tok.piece_to_id[("<special>", "UNK")] == 1
        assert tok.piece_to_id[("<special>", "BOS")] == 2
        assert tok.piece_to_id[("<special>", "EOS")] == 3
        assert tok.piece_to_id[("<special>", "WORD_BOUNDARY")] == 4
        assert tok.piece_to_id[("<special>", "CAP")] == 5
        assert tok.piece_to_id[("<special>", "UPPER")] == 6

    def test_256_byte_tokens_at_7_to_262(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.piece_to_id[("<byte>", 0)] == 7
        assert tok.piece_to_id[("<byte>", 255)] == 262
        byte_ids = [tok.piece_to_id[("<byte>", v)] for v in range(256)]
        assert byte_ids == list(range(7, 263))

    def test_byte_bpe_pieces_then_morpheme_pieces(self, tmp_path):
        # 2 byte merges -> ids 263, 264; morpheme pieces start at 265.
        byte_merges = [("a", "b"), ("ab", "c")]
        tok = _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2 + 4
        )
        assert tok.id_to_piece[263] == "ab"
        assert tok.id_to_piece[264] == "abc"
        assert tok.id_to_piece[265] == MORPH_SEP + "ev"
        assert tok.id_to_piece[266] == MORPH_SEP + "gel"

    def test_pieces_start_in_frequency_order(self, tmp_path):
        tok = _build(tmp_path)
        # No byte merges -> morpheme pieces begin at 263, most frequent first.
        assert tok.id_to_piece[263] == MORPH_SEP + "ev"
        assert tok.id_to_piece[264] == MORPH_SEP + "gel"
        assert tok.id_to_piece[265] == "di"
        assert tok.id_to_piece[266] == "ler"

    def test_piece_to_id_and_id_to_piece_consistent(self, tmp_path):
        tok = _build(tmp_path)
        for tid, piece in enumerate(tok.id_to_piece):
            assert tok.piece_to_id[piece] == tid

    def test_vocab_size_matches_layout(self, tmp_path):
        tok = _build(tmp_path)
        assert tok.vocab_size == _SMALL_VOCAB


# ---------------------------------------------------------------------------------------------
# Encode (morpheme path + fertility invariant)
# ---------------------------------------------------------------------------------------------


class TestEncode:
    def test_fertility_invariant_no_per_word_boundary_token(self, tmp_path, monkeypatch):
        """In-vocab lowercase words cost exactly their piece count (no per-word boundary)."""
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "geldi": {"parsed": True, "morphemes": [{"chunk": "gel"}, {"chunk": "di"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("ev geldi")
        assert len(ids) == 3  # ev -> 1 piece, geldi -> 2 pieces
        assert wb not in ids
        assert ids == [
            tok.piece_to_id[MORPH_SEP + "ev"],
            tok.piece_to_id[MORPH_SEP + "gel"],
            tok.piece_to_id["di"],
        ]

    def test_word_initial_fused_internal_bare(self, tmp_path, monkeypatch):
        analyses = {
            "evler": {"parsed": True, "morphemes": [{"chunk": "ev"}, {"chunk": "ler"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        ids = tok.encode("evler")
        assert ids == [tok.piece_to_id[MORPH_SEP + "ev"], tok.piece_to_id["ler"]]

    def test_oov_word_routes_to_byte_bpe(self, tmp_path, monkeypatch):
        # "zzz" word-initial form "▁zzz" is OOV -> WORD_BOUNDARY + byte-BPE of the surface.
        analyses = {"zzz": {"parsed": True, "morphemes": [{"chunk": "zzz"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        byte_off = tok.piece_to_id[("<byte>", 0)]
        byte_map = bytes_to_unicode()
        ids = tok.encode("zzz")
        expected_bytes = [byte_off + b for b in b"zzz"]
        assert ids[0] == wb
        assert ids[1:] == expected_bytes
        # Sanity: each emitted byte id decodes to the right byte-char.
        assert [byte_map[i - byte_off] for i in ids[1:]] == [byte_map[b] for b in b"zzz"]

    def test_add_bos_eos(self, tmp_path, monkeypatch):
        _install_fake_tr_api(monkeypatch)
        tok = _build(tmp_path)
        bos = tok.piece_to_id[("<special>", "BOS")]
        eos = tok.piece_to_id[("<special>", "EOS")]
        ids = tok.encode("ev", add_bos=True, add_eos=True)
        assert ids[0] == bos and ids[-1] == eos


# ---------------------------------------------------------------------------------------------
# Byte-level BPE fallback
# ---------------------------------------------------------------------------------------------


class TestByteBPEFallback:
    def test_byte_bpe_merges_collapse_bytes_no_single_byte_explosion(
        self, tmp_path, monkeypatch
    ):
        """With byte merges, an OOV word uses byte-BPE pieces, not one token per byte."""
        analyses = {"foo": {"parsed": True, "morphemes": [{"chunk": "foo"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        # Merge f+o, then fo+o -> "foo" becomes a SINGLE byte-BPE piece.
        byte_merges = [("f", "o"), ("fo", "o")]
        tok = _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2 + 4
        )
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("foo")
        # WORD_BOUNDARY + the single merged "foo" byte-BPE piece (1 token, not 3 raw bytes).
        assert ids == [wb, tok.piece_to_id["foo"]]
        assert tok.decode(ids) == "foo"

    def test_byte_bpe_round_trips_exactly(self, tmp_path, monkeypatch):
        analyses = {"foo": {"parsed": True, "morphemes": [{"chunk": "foo"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        byte_merges = [("f", "o"), ("fo", "o")]
        tok = _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2 + 4
        )
        assert tok.decode(tok.encode("foo")) == "foo"

    def test_long_word_uses_byte_bpe_without_analyzer(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        long_word = "a" * 80  # > MAX_WORD_LEN (70)
        ids = tok.encode(long_word)
        assert ids[0] == wb
        assert calls["n"] == 0  # never sent to the analyzer
        assert tok.decode(ids) == long_word


# ---------------------------------------------------------------------------------------------
# Lossless round-trip (THE headline guarantee)
# ---------------------------------------------------------------------------------------------


class TestLosslessRoundTrip:
    def _full_tok(self, tmp_path, monkeypatch, analyses):
        """A tokenizer with a fake tr_api and a few byte merges for compact fallback."""
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        # A handful of byte merges so the fallback isn't pure single-bytes (still lossless).
        byte_merges = [("T", "ü"), ("i", "P")]
        return _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2 + 4
        )

    def test_lowercase_turkish(self, tmp_path, monkeypatch):
        analyses = {"ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]}}
        tok = self._full_tok(tmp_path, monkeypatch, analyses)
        assert tok.decode(tok.encode("ev")) == "ev"

    def test_titlecase_emits_cap_marker(self, tmp_path, monkeypatch):
        # "Ankara" lowercases to "ankara" -> single morpheme "ankara"; vocab has ▁ankara? No.
        # Force it into the vocab so the morpheme path (with CAP) is exercised.
        analyses = {"ankara": {"parsed": True, "morphemes": [{"chunk": "ankara"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        freqs = collections.Counter({MORPH_SEP + "ankara": 100})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=_FIRST_MORPH_ID + 1)
        cap = tok.piece_to_id[("<special>", "CAP")]
        ids = tok.encode("Ankara")
        assert ids[0] == cap
        assert tok.decode(ids) == "Ankara"

    def test_allcaps_emits_upper_marker(self, tmp_path, monkeypatch):
        analyses = {"türkiye": {"parsed": True, "morphemes": [{"chunk": "türkiye"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        freqs = collections.Counter({MORPH_SEP + "türkiye": 100})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=_FIRST_MORPH_ID + 1)
        upper = tok.piece_to_id[("<special>", "UPPER")]
        ids = tok.encode("TÜRKİYE")
        assert ids[0] == upper
        assert tok.decode(ids) == "TÜRKİYE"

    def test_apostrophe_survives_via_byte_bpe(self, tmp_path, monkeypatch):
        # Fake tr_api DROPS the apostrophe (lowercases too): "Türkiye'de" -> "türkiyede".
        # The fidelity check ("".join(pieces) != lowered surface) fails -> byte-BPE the surface.
        analyses = {
            "türkiye'de": {
                "parsed": True,
                "morphemes": [{"chunk": "türkiye"}, {"chunk": "de"}],
            }
        }
        tok = self._full_tok(tmp_path, monkeypatch, analyses)
        text = "Türkiye'de"
        ids = tok.encode(text)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        assert ids[0] == wb  # routed to byte-BPE
        assert tok.decode(ids) == text  # exact, apostrophe + casing preserved

    def test_mixed_case_via_byte_bpe(self, tmp_path, monkeypatch):
        analyses = {"iphone": {"parsed": True, "morphemes": [{"chunk": "iphone"}]}}
        tok = self._full_tok(tmp_path, monkeypatch, analyses)
        ids = tok.encode("iPhone")
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        assert ids[0] == wb  # MIXED -> byte-BPE
        assert tok.decode(ids) == "iPhone"

    def test_foreign_and_emoji_tokens(self, tmp_path, monkeypatch):
        tok = self._full_tok(tmp_path, monkeypatch, analyses={})
        for text in ("café", "naïve", "🎉", "日本語"):
            assert tok.decode(tok.encode(text)) == text

    def test_multi_word_mix(self, tmp_path, monkeypatch):
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "geldi": {"parsed": True, "morphemes": [{"chunk": "gel"}, {"chunk": "di"}]},
            "ankara": {"parsed": True, "morphemes": [{"chunk": "ankara"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        freqs = collections.Counter(
            {
                MORPH_SEP + "ev": 100,
                MORPH_SEP + "gel": 80,
                "di": 60,
                MORPH_SEP + "ankara": 50,
            }
        )
        byte_merges = [("i", "P")]
        tok = _build(
            tmp_path,
            piece_freqs=freqs,
            byte_merges=byte_merges,
            vocab_size=_FIRST_MORPH_ID + 1 + 4,
        )
        text = "ev geldi Ankara iPhone 🎉"
        assert tok.decode(tok.encode(text)) == text

    def test_byte_word_before_morpheme_word(self, tmp_path, monkeypatch):
        # Regression: a byte-BPE (OOV/mixed) word IMMEDIATELY followed by a morpheme word must
        # not feed the ▁-prefixed morpheme piece into the byte decoder (was KeyError '▁').
        # test_multi_word_mix only put byte words last, so it missed this transition.
        analyses = {
            "ev": {"parsed": True, "morphemes": [{"chunk": "ev"}]},
            "geldi": {"parsed": True, "morphemes": [{"chunk": "gel"}, {"chunk": "di"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        freqs = collections.Counter(
            {MORPH_SEP + "ev": 100, MORPH_SEP + "gel": 80, "di": 60}
        )
        tok = _build(
            tmp_path,
            piece_freqs=freqs,
            byte_merges=[("i", "P")],
            vocab_size=_FIRST_MORPH_ID + 1 + 3,
        )
        for text in ("iPhone ev", "🎉 geldi", "iPhone ev geldi 🎉 ev", "café ev"):
            assert tok.decode(tok.encode(text)) == text

    def test_round_trips_multibyte_turkish_via_byte_bpe(self, tmp_path, monkeypatch):
        # "ışığı" lowercases to itself but is OOV (no ▁ışığı in vocab) -> byte-BPE, exact.
        word = "ışığı"
        analyses = {word: {"parsed": True, "morphemes": [{"chunk": word}]}}
        tok = self._full_tok(tmp_path, monkeypatch, analyses)
        assert tok.decode(tok.encode(word)) == word


# ---------------------------------------------------------------------------------------------
# Casing markers
# ---------------------------------------------------------------------------------------------


class TestCasingMarkers:
    def _vocab_with(self, tmp_path, monkeypatch, lower_word):
        analyses = {lower_word: {"parsed": True, "morphemes": [{"chunk": lower_word}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        freqs = collections.Counter({MORPH_SEP + lower_word: 100})
        return _build(tmp_path, piece_freqs=freqs, vocab_size=_FIRST_MORPH_ID + 1)

    def test_lowercase_has_no_marker(self, tmp_path, monkeypatch):
        tok = self._vocab_with(tmp_path, monkeypatch, "ankara")
        cap = tok.piece_to_id[("<special>", "CAP")]
        upper = tok.piece_to_id[("<special>", "UPPER")]
        ids = tok.encode("ankara")
        assert cap not in ids and upper not in ids

    def test_titlecase_marker_only_for_title(self, tmp_path, monkeypatch):
        tok = self._vocab_with(tmp_path, monkeypatch, "ankara")
        cap = tok.piece_to_id[("<special>", "CAP")]
        assert tok.encode("Ankara")[0] == cap

    def test_allcaps_marker_only_for_allcaps(self, tmp_path, monkeypatch):
        tok = self._vocab_with(tmp_path, monkeypatch, "türkiye")
        upper = tok.piece_to_id[("<special>", "UPPER")]
        assert tok.encode("TÜRKİYE")[0] == upper

    def test_marker_restored_in_decode(self, tmp_path, monkeypatch):
        tok = self._vocab_with(tmp_path, monkeypatch, "ankara")
        assert tok.decode(tok.encode("Ankara")) == "Ankara"


# ---------------------------------------------------------------------------------------------
# Fidelity check
# ---------------------------------------------------------------------------------------------


class TestFidelityCheck:
    def test_pieces_not_concatenating_back_route_to_byte_bpe(self, tmp_path, monkeypatch):
        # Fake tr_api returns pieces that DON'T join back to the lowercased surface
        # (a dropped char). The fidelity check must reject the morpheme path -> byte-BPE.
        analyses = {
            "abcd": {"parsed": True, "morphemes": [{"chunk": "ab"}, {"chunk": "d"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("abcd")
        assert ids[0] == wb  # fidelity check failed -> byte-BPE fallback
        assert tok.decode(ids) == "abcd"

    def test_oov_piece_routes_to_byte_bpe(self, tmp_path, monkeypatch):
        # Pieces concatenate back, but an internal piece is OOV -> byte-BPE the whole surface.
        analyses = {
            "evxx": {"parsed": True, "morphemes": [{"chunk": "ev"}, {"chunk": "xx"}]},
        }
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        tok = _build(tmp_path)
        wb = tok.piece_to_id[("<special>", "WORD_BOUNDARY")]
        ids = tok.encode("evxx")
        assert ids[0] == wb
        assert tok.decode(ids) == "evxx"


# ---------------------------------------------------------------------------------------------
# Fast table
# ---------------------------------------------------------------------------------------------


class TestFastTable:
    def test_tabled_word_skips_analyzer(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        # Table is keyed by the LOWERCASED word; stores BARE pieces.
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        ids = tok.encode("evler")
        assert ids == [tok.piece_to_id[MORPH_SEP + "ev"], tok.piece_to_id["ler"]]
        assert calls["n"] == 0  # the table satisfied the lookup, analyzer untouched

    def test_titlecase_tabled_word_lowercased_lookup(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        _install_fake_tr_api(monkeypatch, calls=calls)
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        cap = tok.piece_to_id[("<special>", "CAP")]
        ids = tok.encode("Evler")  # lowercases to "evler" -> table hit
        assert ids[0] == cap
        assert calls["n"] == 0
        assert tok.decode(ids) == "Evler"

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
            byte_merges=[],
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
        tok = _build(tmp_path, merges=[["ev", "ler"]], byte_merges=[("a", "b")],
                     vocab_size=_FIRST_MORPH_ID + 1 + 4)
        out = tmp_path / "saved"
        tok.save(str(out))
        loaded = MorphemeTokenizer.load(str(out))
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.id_to_piece == tok.id_to_piece
        assert loaded.piece_to_id == tok.piece_to_id
        # Both merge engines are rebuilt and behave identically.
        assert loaded._bpe.encode(["ev", "ler"]) == MorphemeBPE([("ev", "ler")]).encode(
            ["ev", "ler"]
        )
        assert loaded._byte_merges == [("a", "b")]

    def test_round_trips_byte_merges_and_fallback(self, tmp_path, monkeypatch):
        byte_merges = [("f", "o"), ("fo", "o")]
        tok = _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2 + 4
        )
        out = tmp_path / "saved"
        tok.save(str(out))
        loaded = MorphemeTokenizer.load(str(out))
        analyses = {"foo": {"parsed": True, "morphemes": [{"chunk": "foo"}]}}
        _install_fake_tr_api(monkeypatch, analyses=analyses)
        assert loaded.decode(loaded.encode("foo")) == "foo"

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
        tok = _build(tmp_path, table={"evler": ["ev", "ler"]})
        out = tmp_path / "saved"
        tok.save(str(out))

        def _boom(repo_path=None):
            raise AssertionError("tr_api should not be needed for tabled/saved-load path")

        monkeypatch.setattr("turkish_corpus.morphology.ensure_tr_api_importable", _boom)

        loaded = MorphemeTokenizer.load(str(out))  # must not touch tr_api
        ids = loaded.encode("evler")  # tabled -> still no tr_api
        assert ids == [loaded.piece_to_id[MORPH_SEP + "ev"], loaded.piece_to_id["ler"]]
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
        vocab_size = _FIRST_MORPH_ID + 2  # room for only 2 morpheme pieces
        tok = _build(tmp_path, vocab_size=vocab_size)
        assert tok.vocab_size == vocab_size
        assert tok.id_to_piece[_FIRST_MORPH_ID] == MORPH_SEP + "ev"
        assert tok.id_to_piece[_FIRST_MORPH_ID + 1] == MORPH_SEP + "gel"
        assert "di" not in tok.piece_to_id
        assert "ler" not in tok.piece_to_id

    def test_byte_merges_consume_id_space(self, tmp_path):
        # 3 byte merges + 2 morpheme pieces.
        byte_merges = [("a", "b"), ("c", "d"), ("e", "f")]
        tok = _build(
            tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 3 + 2
        )
        # Byte-BPE pieces fill 263..265; morpheme pieces 266..267.
        assert tok.id_to_piece[263:266] == ["ab", "cd", "ef"]
        assert tok.id_to_piece[266] == MORPH_SEP + "ev"
        assert tok.id_to_piece[267] == MORPH_SEP + "gel"
        assert "di" not in tok.piece_to_id

    def test_frequency_ordering(self, tmp_path):
        freqs = collections.Counter({"rare": 1, "common": 1000, "mid": 50})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=_FIRST_MORPH_ID + 3)
        assert tok.id_to_piece[263:266] == ["common", "mid", "rare"]

    def test_tie_broken_by_piece_string(self, tmp_path):
        freqs = collections.Counter({"b": 10, "a": 10, "c": 10})
        tok = _build(tmp_path, piece_freqs=freqs, vocab_size=_FIRST_MORPH_ID + 3)
        assert tok.id_to_piece[263:266] == ["a", "b", "c"]

    def test_rejects_too_small_vocab(self, tmp_path):
        with pytest.raises(ValueError, match="too small"):
            _build(tmp_path, vocab_size=_FIRST_MORPH_ID)  # no room for morpheme pieces

    def test_rejects_too_small_vocab_with_byte_merges(self, tmp_path):
        byte_merges = [("a", "b"), ("c", "d")]
        with pytest.raises(ValueError, match="too small"):
            _build(tmp_path, byte_merges=byte_merges, vocab_size=_FIRST_MORPH_ID + 2)

    def test_default_vocab_size_is_64000(self):
        assert DEFAULT_VOCAB_SIZE == 64_000
