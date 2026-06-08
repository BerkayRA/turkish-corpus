# Turkish-specific gotchas

The traps that generic (English-calibrated) corpus pipelines get wrong. Each is handled by
this package; this doc explains *why*, so the behavior isn't "simplified" back into a bug.

## 1. The dotless/dotted-i casing problem (the #1 silent corruptor)

Turkish has **four** i-letters in two case pairs that violate default Unicode casing:

| upper | lower | name |
|-------|-------|------|
| `I` (U+0049) | `ı` (U+0131) | dotless i |
| `İ` (U+0130) | `i` (U+0069) | dotted i |

Python's defaults are wrong for Turkish:

```python
"I".lower()    # 'i'   — WRONG for Turkish, should be 'ı'
"İ".lower()    # 'i̇'   — TWO characters: 'i' + U+0307 combining dot
"KISA".lower() # 'kisa' — WRONG, should be 'kısa' ("short")
```

**Why it silently corrupts a pipeline.** Case-normalization is a standard pre-hash step
for deduplication. With default casing, `"KISA"` and the unrelated word `"kisa"` either
collide (false duplicate → real content discarded) or fail to match a near-duplicate
(false non-duplicate), degrading MinHash precision and recall. It also breaks stop-word
and bad-word matching.

**Our fix** (`normalization.turkish_lower` / `turkish_upper`): correct `tr`-locale casing
that maps `I→ı`, `İ→i` (and the reverse), handles the decomposed `I`+combining-dot form,
and **preserves the `ı`/`i` distinction** — they are distinct letters and must never be
folded together. Pure-Python by default; opt into ICU with `--extra icu` for full
Unicode-spec coverage of rare edge cases.

> We do **not** lowercase the stored corpus text — a model needs case. Casing is applied
> only to build dedup keys (`normalize_for_dedup`) and to match word lists.

## 2. Decomposed combining marks / mixed normalization

`İ` (U+0130) and `I`+`U+0307` render identically but hash differently. We NFC-normalize so
each grapheme has one canonical codepoint, stabilizing dedup hashes and tokenization.
`normalize_text` also strips zero-width and bidi formatting characters (ZWSP, ZWNJ, LRM,
BOM, …) that frequently survive web extraction.

## 3. Diacritics and ASCII-degraded Turkish

A lot of informal/older Turkish was written in ASCII ("Turkce" for "Türkçe", "ogrenci" for
"öğrenci"). This lowers language-ID confidence (risk of wrongful discard) and breaks
morphological analysis. Strategy: detect and route ASCII-degraded text rather than drop
it, and optionally restore diacritics (deasciification) on high-confidence segments.
Encoding/mojibake repair is available via `--extra encoding` (ftfy); HPLT v2 is already
ftfy-cleaned, so this matters mainly for freshly crawled text.

## 4. Agglutination breaks English-tuned filter thresholds

Turkish is agglutinative — one stem yields long surface forms
(`ev → evler → evlerimizden → evlerimizdenmiş`). Consequences:

- **Mean word length.** Gopher's English 3.0–10.0 character band rejects normal Turkish.
  We widen the upper bound to 12.0 (`filters.TurkishQualityThresholds`).
- **Stop-word checks.** Gopher's "≥2 stop words" / stop-word-ratio require a *Turkish*
  list. We ship `filters.TURKISH_STOPWORDS` (curated starter; derive a fuller list from
  Turkish-Wikipedia word frequency for production, per FineWeb-2's method).
- **Dedup recall.** Word-level n-gram MinHash sees inflected variants as distinct, so
  near-duplicate *paraphrases* are under-detected. We keep MinHash (FineWeb-2's 14×8,
  5-gram config) but don't expect English-level near-dup recall; lemma/stem-aware
  shingling is a future enhancement.

## 5. KVKK: Turkish-specific PII

Turkey's KVKK (Law 6698) is GDPR-aligned, extraterritorial, and treats automated
collection as processing. datatrove's `PIIFormatter` only knows emails/IPs. We add
(`pii.py`):

- **T.C. Kimlik No** — 11-digit national ID; we **validate the checksum** before redacting
  so unrelated 11-digit numbers (years, codes, ISBNs) survive.
- **Turkish phone numbers** — `+90`/`0090`/trunk-`0`, mobile and landline formats.
- **TR IBAN** — `TR` + 24 digits, **mod-97 validated**.
- Vehicle plates — available but **off by default** (high false-positive rate against
  ordinary "<province> <word> <number>" prose).

## 6. Tokenizer fertility

Multilingual tokenizers over-fragment Turkish (a "tokenization premium" up to ~10–15× more
tokens/word than English). Two implications: (a) corpus token *counts* depend heavily on
the tokenizer, so size the budget with **your morphology-aware tokenizer**
(`tokenizer.load_token_counter`), not a generic one; (b) measure fertility as
`hf_count / whitespace_count` to quantify how much better your tokenizer is on Turkish.
