"""Turkish-aware Unicode normalization and casing.

This module exists because *generic* text pipelines silently corrupt Turkish at two
points, and both are subtle enough to pass code review and unit tests written by an
English speaker:

1. The dotless/dotted-i problem. Turkish has FOUR i-letters in two case pairs that
   violate the default Unicode casing rules:

       I  (U+0049) <-> ı (U+0131, dotless small i)
       İ  (U+0130, dotted capital I) <-> i (U+0069)

   Python's ``str.lower()`` maps ``I -> i`` (wrong for Turkish: it must be ``ı``) and
   ``"İ".lower()`` even returns a TWO-character string (``i`` + U+0307 combining dot).
   When a pipeline lowercases text before hashing for deduplication, this makes
   ``"KISA"`` ("short", correctly ``"kısa"``) collide with the unrelated ``"kisa"`` —
   producing false duplicates and destroying MinHash precision. Crucially, ``ı`` and
   ``i`` are DISTINCT LETTERS in Turkish; we must preserve that distinction, never fold
   one into the other.

2. Decomposed combining marks. ``İ`` and ``I + U+0307`` look identical but hash
   differently. We NFC-normalize so each grapheme has a single canonical codepoint,
   which stabilizes dedup hashes and downstream tokenization.

Design notes
------------
- We DO NOT lowercase the corpus text itself: a language model needs casing. Casing is
  applied only to build dedup keys and to match stopword / bad-word lists. Corpus-text
  normalization (:func:`normalize_text`) is NFC + optional encoding repair + control /
  whitespace cleanup, and preserves case.
- A correct pure-Python Turkish casing is implemented here so the package works with
  zero native dependencies. If PyICU is installed (``uv sync --extra icu``) you can opt
  into ICU's ``tr`` locale casing via ``use_icu=True`` for full Unicode-spec coverage of
  rare edge cases.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "nfc",
    "turkish_lower",
    "turkish_upper",
    "normalize_for_dedup",
    "normalize_text",
    "fix_encoding",
]

# Combining dot above — the mark that turns I into a dotted i under NFD.
_COMBINING_DOT_ABOVE = "̇"

# Turkish special-case casing pairs. Every other letter casings identically to the
# locale-independent default, so we only special-case the i-family here.
_TR_LOWER_MAP = str.maketrans(
    {
        "I": "ı",  # dotless: capital I -> dotless small i
        "İ": "i",  # dotted: dotted capital I -> dotted small i
    }
)
_TR_UPPER_MAP = str.maketrans(
    {
        "i": "İ",  # dotted small i -> dotted capital I
        "ı": "I",  # dotless small i -> capital I
    }
)

_WS_RUN_RE = re.compile(r"[^\S\n]+")  # runs of horizontal whitespace
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)  # line-trailing spaces
_BLANKLINES_RE = re.compile(r"\n{3,}")  # 3+ newlines -> 2

# Whitespace preserved while every other control/format character is stripped.
_KEEP_WHITESPACE = frozenset("\t\n\r")


def nfc(text: str) -> str:
    """Return ``text`` in Unicode Normalization Form C (canonical composition)."""
    return unicodedata.normalize("NFC", text)


def turkish_lower(text: str, *, use_icu: bool = False) -> str:
    """Lowercase ``text`` using Turkish (``tr``) rules.

    ``I -> ı`` and ``İ -> i``; all other characters lowercase normally. ``ı`` and ``i``
    remain distinct. Handles both composed ``İ`` (U+0130) and the decomposed
    ``I`` + combining-dot sequence.
    """
    if use_icu:
        return _icu_case(text, lower=True)
    # Collapse the decomposed dotted-I sequence to dotted small i first, so it does not
    # fall through to the ``I -> ı`` rule below.
    text = text.replace("I" + _COMBINING_DOT_ABOVE, "i")
    # Apply the Turkish i-family special cases, then lowercase the remainder. The
    # already-substituted ``ı``/``i`` are unaffected by ``str.lower()``.
    return text.translate(_TR_LOWER_MAP).lower()


def turkish_upper(text: str, *, use_icu: bool = False) -> str:
    """Uppercase ``text`` using Turkish (``tr``) rules.

    ``i -> İ`` and ``ı -> I``; all other characters uppercase normally.
    """
    if use_icu:
        return _icu_case(text, lower=False)
    return text.translate(_TR_UPPER_MAP).upper()


def normalize_for_dedup(text: str, *, use_icu: bool = False) -> str:
    """Produce a stable key for (near-)duplicate detection.

    NFC + Turkish lowercase + horizontal-whitespace collapse. Preserves the ``ı``/``i``
    distinction; only intended for hashing/comparison, never for storage.
    """
    text = nfc(text)
    text = turkish_lower(text, use_icu=use_icu)
    text = _WS_RUN_RE.sub(" ", text)
    return text.strip()


def normalize_text(text: str, *, fix_mojibake: bool = False) -> str:
    """Normalize corpus text for storage while PRESERVING case.

    Steps: optional mojibake repair (ftfy) -> NFC -> strip control/zero-width chars ->
    collapse horizontal whitespace runs -> cap consecutive blank lines at 2 -> strip.
    """
    if fix_mojibake:
        text = fix_encoding(text)
    text = nfc(text)
    text = _strip_controls_and_formats(text)
    text = _WS_RUN_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _strip_controls_and_formats(text: str) -> str:
    """Remove all Unicode control (Cc) and format (Cf) characters except tab/newline/CR.

    Category-based rather than an enumerated blocklist, so it also removes the SOFT HYPHEN
    (U+00AD), invisible-math operators (U+2061-2064), bidi isolates (U+2066-2069), BOM,
    and zero-width characters. These survive HTML extraction and, left in place, let PII
    like ``1000<soft-hyphen>0000146`` slip past the redactors (a KVKK leak vector).
    """
    if not text:
        return text
    return "".join(
        ch
        for ch in text
        if ch in _KEEP_WHITESPACE or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def fix_encoding(text: str) -> str:
    """Repair mojibake using ftfy if available; otherwise return ``text`` unchanged.

    HPLT v2 is already ftfy-cleaned, so this is mainly for freshly crawled text. Install
    with ``uv sync --extra encoding``.
    """
    try:
        import ftfy  # noqa: PLC0415  (optional, lazily imported)
    except ImportError:
        return text
    return ftfy.fix_text(text)


def _icu_case(text: str, *, lower: bool) -> str:
    """ICU ``tr_TR`` casing. Requires PyICU (``uv sync --extra icu``)."""
    try:
        import icu  # noqa: PLC0415  (optional, lazily imported)
    except ImportError as exc:  # pragma: no cover - exercised only without PyICU
        raise RuntimeError(
            "use_icu=True requires PyICU. Install with `uv sync --extra icu` "
            "(needs system icu4c: `brew install icu4c`)."
        ) from exc
    locale = icu.Locale("tr_TR")
    us = icu.UnicodeString(text)
    return str(us.toLower(locale) if lower else us.toUpper(locale))
