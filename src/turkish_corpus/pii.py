"""Turkish-specific PII detection and redaction for KVKK compliance.

Turkey's KVKK (Law No. 6698) is GDPR-aligned, applies extraterritorially to data about
Turkish residents, and treats automated collection as "processing", with material
administrative and criminal penalties. A pretraining corpus must therefore scrub
personal data. datatrove's built-in ``PIIFormatter`` handles emails and IP addresses;
this module adds the Turkish-specific identifiers it does not know about:

- T.C. Kimlik No (national ID): an 11-digit number with a verifiable checksum. We
  validate the checksum before redacting so we do not destroy unrelated 11-digit
  numbers (years-as-digits, ISBNs, etc.).
- Turkish phone numbers (mobile +90 5xx and landline formats).
- Turkish IBAN (``TR`` + 24 digits), optionally validated with the ISO 7064 mod-97 check.
- Vehicle plates (optional; off by default — high false-positive rate).

Everything here is pure-Python and dependency-free so it is unit-testable without the
heavy ``pipeline`` extra.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# Protocol-defined lengths.
_TC_KIMLIK_LENGTH = 11
_TR_IBAN_LENGTH = 26

__all__ = [
    "validate_tc_kimlik",
    "validate_tr_iban",
    "redact_turkish_pii",
    "RedactionResult",
    "PLACEHOLDERS",
]

PLACEHOLDERS = {
    "tc_kimlik": "<TC_KIMLIK>",
    "phone": "<TELEFON>",
    "iban": "<IBAN>",
    "plate": "<PLAKA>",
}

# 11 digits, not preceded/followed by another digit. Validated by checksum afterwards.
_TC_CANDIDATE_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")

# Turkish IBAN: TR + 2 check digits + 22 alphanumerics, validated by length + mod-97
# afterwards. Allow space- or dash-separated groups of four. We anchor with digit-only
# lookaround (not \b) so an IBAN abutting a label letter ("hesapTR33...") still matches,
# while a leading/trailing digit (part of a longer number) does not.
_IBAN_RE = re.compile(
    r"(?<!\d)TR\d{2}(?:[ \-]?[A-Z0-9]{4}){5}[ \-]?[A-Z0-9]{2}(?!\w)", re.IGNORECASE
)

# Turkish phone numbers. To avoid matching arbitrary 10-digit numbers (invoice/order ids,
# years), we require a valid leading group: an international/trunk prefix followed by a
# mobile (5xx) or landline (2xx-4xx) code, OR a bare mobile 5xx. Optional parentheses
# around the code; separators may be spaces, dots, or hyphens.
_PHONE_RE = re.compile(
    r"""
    (?<!\w)
    (?:
        (?:(?:\+|00)90|0)[\s.\-]?\(?(?:5\d{2}|[2-4]\d{2})\)?  # (intl 90 | trunk 0) + code
      | 5\d{2}                                                # bare mobile, no prefix
    )
    [\s.\-]?\d{3}
    [\s.\-]?\d{2}
    [\s.\-]?\d{2}
    (?!\w)
    """,
    re.VERBOSE,
)

# Turkish plate: 2-digit province (01-81) + 1-3 letters + 2-4 digits, optional spaces.
_PLATE_RE = re.compile(
    r"\b(0[1-9]|[1-7][0-9]|8[01])[\s]?[A-ZÇĞİÖŞÜ]{1,3}[\s]?\d{2,4}\b", re.IGNORECASE
)


@dataclass
class RedactionResult:
    """Outcome of :func:`redact_turkish_pii`."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def validate_tc_kimlik(value: str) -> bool:
    """Return True if ``value`` is a checksum-valid T.C. Kimlik No.

    Rules: 11 digits; first digit non-zero;
      d10 = ((d1+d3+d5+d7+d9) * 7 - (d2+d4+d6+d8)) mod 10;
      d11 = (d1+...+d10) mod 10.
    """
    if not isinstance(value, str):
        return False
    if not (len(value) == _TC_KIMLIK_LENGTH and value.isdigit()):
        return False
    d = [int(c) for c in value]
    if d[0] == 0:
        return False
    odd_sum = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    if (odd_sum * 7 - even_sum) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def validate_tr_iban(value: str) -> bool:
    """Validate a Turkish IBAN with the ISO 7064 mod-97 check (spaces ignored)."""
    if not isinstance(value, str):
        return False
    compact = value.replace(" ", "").replace("-", "").upper()
    if len(compact) != _TR_IBAN_LENGTH or not compact.startswith("TR"):
        return False
    rearranged = compact[4:] + compact[:4]
    # Map letters to numbers (A=10 ... Z=35) per the IBAN algorithm.
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def redact_turkish_pii(
    text: str,
    *,
    redact_tc: bool = True,
    redact_phone: bool = True,
    redact_iban: bool = True,
    redact_plate: bool = False,
    validate_iban: bool = True,
) -> RedactionResult:
    """Redact Turkish PII from ``text``, returning the cleaned text and per-type counts.

    Order matters: IBAN (most specific) before phone before T.C. Kimlik, so a phone-like
    run inside an IBAN is not partially redacted first. Plates are off by default because
    the pattern collides with ordinary "<province> <word> <number>" prose.
    """
    counts: dict[str, int] = {}

    if redact_iban:
        text, n = _sub_validated(_IBAN_RE, text, PLACEHOLDERS["iban"],
                                 validate_tr_iban if validate_iban else None)
        if n:
            counts["iban"] = n

    if redact_phone:
        text, n = _subn(_PHONE_RE, text, PLACEHOLDERS["phone"])
        if n:
            counts["phone"] = n

    if redact_tc:
        # Only redact 11-digit runs that pass the checksum.
        text, n = _sub_validated(_TC_CANDIDATE_RE, text, PLACEHOLDERS["tc_kimlik"],
                                 validate_tc_kimlik)
        if n:
            counts["tc_kimlik"] = n

    if redact_plate:
        text, n = _subn(_PLATE_RE, text, PLACEHOLDERS["plate"])
        if n:
            counts["plate"] = n

    return RedactionResult(text=text, counts=counts)


def _subn(pattern: re.Pattern[str], text: str, placeholder: str) -> tuple[str, int]:
    return pattern.subn(placeholder, text)


def _sub_validated(
    pattern: re.Pattern[str],
    text: str,
    placeholder: str,
    validator: Callable[[str], bool] | None,
) -> tuple[str, int]:
    """Replace matches with ``placeholder`` only when ``validator`` accepts the match.

    ``validator`` receives the raw matched substring. If ``validator`` is None, all
    matches are replaced.
    """
    count = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal count
        matched = m.group(0)
        if validator is None or validator(matched):
            count += 1
            return placeholder
        return matched

    return pattern.sub(_repl, text), count
