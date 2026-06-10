"""Tests for Turkish-aware normalization and casing.

These pin the exact behavior that generic pipelines get wrong: the dotless/dotted-i
casing and the preservation of the ı/i distinction.
"""

import unicodedata

import pytest

from turkish_corpus.normalization import (
    fix_encoding,
    nfc,
    normalize_for_dedup,
    normalize_text,
    turkish_lower,
    turkish_title,
    turkish_upper,
)


class TestTurkishLower:
    def test_capital_dotless_i_maps_to_dotless_lower(self):
        # "KISA" ("short") must lowercase to "kısa", NOT "kisa".
        assert turkish_lower("KISA") == "kısa"

    def test_capital_dotted_i_maps_to_dotted_lower(self):
        assert turkish_lower("İSTANBUL") == "istanbul"

    def test_mixed_word_both_i_kinds(self):
        assert turkish_lower("DİYARBAKIR") == "diyarbakır"

    def test_preserves_i_dotless_distinction(self):
        # The whole point: I-> ı and İ -> i must NOT collapse together.
        assert turkish_lower("KISA") != turkish_lower("KİSA")
        assert turkish_lower("KİSA") == "kisa"

    def test_differs_from_default_python_lower(self):
        # Guards against someone "simplifying" this to str.lower().
        assert turkish_lower("I") == "ı"
        assert "I".lower() == "i"

    def test_decomposed_dotted_capital_i(self):
        # "I" + combining dot above should behave like İ -> i, not I -> ı.
        decomposed = "I" + "̇" + "GNE"  # İGNE in NFD-ish form
        assert turkish_lower(decomposed) == "igne"

    def test_non_turkish_letters_lower_normally(self):
        assert turkish_lower("ABC ÇĞÖŞÜ") == "abc çğöşü"


class TestTurkishTitle:
    def test_dotless_lowercase_i_titlecases_to_dotless_capital(self):
        # "ışık" must titlecase to "Işık" (dotless I), not "İşık".
        assert turkish_title("ışık") == "Işık"

    def test_dotted_lowercase_i_titlecases_to_dotted_capital(self):
        assert turkish_title("istanbul") == "İstanbul"

    def test_rest_is_lowercased(self):
        assert turkish_title("ANKARA") == "Ankara"

    def test_empty_string(self):
        assert turkish_title("") == ""

    def test_round_trips_with_lower(self):
        # title(lower(Titlecased)) == Titlecased for a normal capitalized word.
        assert turkish_title(turkish_lower("Ankara")) == "Ankara"


class TestTurkishUpper:
    def test_dotted_i_uppercases_to_dotted_capital(self):
        assert turkish_upper("iğne") == "İĞNE"

    def test_dotless_i_uppercases_to_plain_capital(self):
        assert turkish_upper("ışık") == "IŞIK"

    def test_roundtrip_i_family(self):
        for word in ("istanbul", "ışık", "iğne", "kısa"):
            assert turkish_lower(turkish_upper(word)) == word


class TestNfc:
    def test_composes_decomposed_dotted_capital_i(self):
        decomposed = "İ"
        assert nfc(decomposed) == "İ"
        assert unicodedata.is_normalized("NFC", nfc(decomposed))


class TestNormalizeForDedup:
    def test_collapses_whitespace_and_lowercases_turkish(self):
        assert normalize_for_dedup("KISA   metin\t  ") == "kısa metin"

    def test_distinct_words_stay_distinct(self):
        assert normalize_for_dedup("KISA") != normalize_for_dedup("KISA".replace("I", "İ"))


class TestNormalizeText:
    def test_preserves_case(self):
        assert normalize_text("İstanbul'da Bir Gün") == "İstanbul'da Bir Gün"

    def test_strips_zero_width_chars(self):
        assert normalize_text("a​b‌‍c") == "abc"

    def test_strips_soft_hyphen(self):
        # SOFT HYPHEN (U+00AD) is rampant in scraped HTML and must not survive — left in
        # place it lets obfuscated PII (1000\xad0000146) defeat redaction.
        assert normalize_text("1000­0000146") == "10000000146"

    def test_strips_invisible_math_and_bidi_isolates(self):
        # invisible separator (U+2063) and bidi isolate (U+2066) + pop (U+2069)
        assert normalize_text("a⁣b⁦c⁩d") == "abcd"

    def test_strips_other_control_chars(self):
        assert normalize_text("a\x00b\x07c") == "abc"

    def test_strips_bom_and_control_but_keeps_newlines(self):
        assert normalize_text("﻿satır1\nsatır2") == "satır1\nsatır2"

    def test_collapses_horizontal_whitespace_only(self):
        assert normalize_text("a    b\nc") == "a b\nc"

    def test_caps_consecutive_blank_lines(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_applies_nfc(self):
        assert normalize_text("İstanbul") == "İstanbul"

    def test_strips_trailing_line_spaces(self):
        assert normalize_text("satır   \nbaşka   ") == "satır\nbaşka"

    def test_fix_mojibake_branch_is_safe_without_ftfy(self):
        # With ftfy absent, fix_mojibake must be a no-op rather than crash.
        assert normalize_text("İstanbul", fix_mojibake=True) == "İstanbul"


class TestFixEncoding:
    def test_returns_unchanged_when_ftfy_missing(self):
        # ftfy is an optional extra; the fallback must return input verbatim.
        try:
            import ftfy  # noqa: F401
        except ImportError:
            assert fix_encoding("Ã§ÄŸ") == "Ã§ÄŸ"
        else:  # pragma: no cover - only when the encoding extra is installed
            assert isinstance(fix_encoding("Ã§ÄŸ"), str)


def test_icu_flag_raises_without_pyicu_if_missing():
    try:
        import icu  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="PyICU"):
            turkish_lower("KISA", use_icu=True)
    else:  # pragma: no cover - only when PyICU is installed
        assert turkish_lower("KISA", use_icu=True) == "kısa"
