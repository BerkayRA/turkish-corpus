"""Tests for Turkish PII detection/redaction (KVKK)."""

from turkish_corpus.normalization import normalize_text
from turkish_corpus.pii import (
    PLACEHOLDERS,
    redact_turkish_pii,
    validate_tc_kimlik,
    validate_tr_iban,
)

# Checksum-valid synthetic T.C. Kimlik No (generated to satisfy the algorithm, not a
# real person's number).
VALID_TC = "10000000146"
# Well-known valid Turkish IBAN example.
VALID_IBAN = "TR330006100519786457841326"


class TestValidateTcKimlik:
    def test_accepts_valid(self):
        assert validate_tc_kimlik(VALID_TC) is True

    def test_rejects_bad_checksum(self):
        assert validate_tc_kimlik("10000000140") is False

    def test_rejects_leading_zero(self):
        assert validate_tc_kimlik("0" + VALID_TC[1:]) is False

    def test_rejects_wrong_length(self):
        assert validate_tc_kimlik("123456789") is False

    def test_rejects_non_digits(self):
        assert validate_tc_kimlik("1000000014x") is False


class TestValidateTrIban:
    def test_accepts_valid(self):
        assert validate_tr_iban(VALID_IBAN) is True

    def test_accepts_valid_with_spaces(self):
        spaced = "TR33 0006 1005 1978 6457 8413 26"
        assert validate_tr_iban(spaced) is True

    def test_rejects_bad_check_digits(self):
        assert validate_tr_iban("TR000006100519786457841326") is False

    def test_rejects_wrong_length(self):
        assert validate_tr_iban("TR3300061005197864578413") is False

    def test_accepts_dash_separated(self):
        assert validate_tr_iban("TR33-0006-1005-1978-6457-8413-26") is True


class TestNoneGuards:
    def test_tc_validator_handles_none(self):
        assert validate_tc_kimlik(None) is False  # type: ignore[arg-type]

    def test_iban_validator_handles_none(self):
        assert validate_tr_iban(None) is False  # type: ignore[arg-type]


class TestRedactTcKimlik:
    def test_redacts_valid_tc(self):
        result = redact_turkish_pii(f"Kimlik numaram {VALID_TC} olarak kayıtlı.")
        assert PLACEHOLDERS["tc_kimlik"] in result.text
        assert VALID_TC not in result.text
        assert result.counts.get("tc_kimlik") == 1

    def test_leaves_invalid_11_digit_numbers(self):
        # An 11-digit run that fails the checksum must NOT be redacted as a TC number.
        text = "Sipariş kodu 12345678901 numaralı."
        result = redact_turkish_pii(text, redact_phone=False)
        assert "12345678901" in result.text
        assert "tc_kimlik" not in result.counts


class TestRedactPhone:
    def test_redacts_intl_mobile(self):
        result = redact_turkish_pii("Beni +90 532 123 45 67 numarasından ara.")
        assert PLACEHOLDERS["phone"] in result.text
        assert result.counts.get("phone") == 1

    def test_redacts_national_landline(self):
        result = redact_turkish_pii("Ofis: 0212 123 45 67")
        assert PLACEHOLDERS["phone"] in result.text

    def test_redacts_bare_mobile_without_prefix(self):
        result = redact_turkish_pii("532 123 45 67")
        assert PLACEHOLDERS["phone"] in result.text

    def test_does_not_match_arbitrary_10_digit_number(self):
        # An invoice/order/year code must NOT be redacted as a phone number.
        text = "Fatura no 1234567890 ve sipariş 2023123456."
        result = redact_turkish_pii(text, redact_tc=False)
        assert "1234567890" in result.text
        assert "2023123456" in result.text
        assert "phone" not in result.counts


class TestRedactIban:
    def test_redacts_valid_iban(self):
        result = redact_turkish_pii(f"IBAN: {VALID_IBAN}")
        assert PLACEHOLDERS["iban"] in result.text
        assert VALID_IBAN not in result.text
        assert result.counts.get("iban") == 1

    def test_redacts_iban_abutting_label_letter(self):
        # "hesapTR33..." (no separator) must still be caught.
        result = redact_turkish_pii(f"hesap{VALID_IBAN}", redact_phone=False)
        assert PLACEHOLDERS["iban"] in result.text

    def test_redacts_dash_separated_iban(self):
        spaced = "TR33-0006-1005-1978-6457-8413-26"
        result = redact_turkish_pii(f"IBAN: {spaced}", redact_phone=False)
        assert PLACEHOLDERS["iban"] in result.text

    def test_skips_validation_when_disabled(self):
        bad_iban = "TR000006100519786457841326"  # fails mod-97
        result = redact_turkish_pii(f"IBAN: {bad_iban}", validate_iban=False,
                                    redact_phone=False)
        assert PLACEHOLDERS["iban"] in result.text


class TestRedactPlate:
    def test_redacts_plate_when_enabled(self):
        result = redact_turkish_pii("Araç plakası 34 ABC 123 kayıtlı.", redact_plate=True)
        assert PLACEHOLDERS["plate"] in result.text
        assert result.counts.get("plate") == 1

    def test_redacts_lowercase_plate(self):
        result = redact_turkish_pii("plaka 06 ab 1234", redact_plate=True,
                                    redact_phone=False)
        assert PLACEHOLDERS["plate"] in result.text


class TestNormalizeThenRedact:
    def test_soft_hyphen_obfuscated_tc_is_caught_after_normalization(self):
        # The realistic KVKK leak vector: a soft hyphen splitting a TC number. The
        # pipeline runs TurkishNormalizer before TurkishPIIRedactor, so normalization
        # must de-obfuscate it first.
        obfuscated = "1000­0000146"
        cleaned = normalize_text(obfuscated)
        result = redact_turkish_pii(cleaned)
        assert PLACEHOLDERS["tc_kimlik"] in result.text
        assert VALID_TC not in result.text


class TestRedactionResult:
    def test_total_sums_counts(self):
        text = f"TC {VALID_TC}, tel +90 532 123 45 67, {VALID_IBAN}"
        result = redact_turkish_pii(text)
        assert result.total == sum(result.counts.values())
        assert result.total >= 3

    def test_clean_text_unchanged(self):
        text = "Bu cümlede kişisel veri yok."
        result = redact_turkish_pii(text)
        assert result.text == text
        assert result.total == 0

    def test_plate_off_by_default(self):
        # Ordinary prose that matches the plate pattern must survive by default.
        result = redact_turkish_pii("34 ABC 123 gibi bir ifade")
        assert "plate" not in result.counts
