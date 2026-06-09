# Filter calibration from Turkish Wikipedia

The corpus-quality filters (Gopher-style word-count / mean-word-length / symbol / non-alpha
checks, plus a stop-word presence check) were calibrated on **English**. Two of them are
distorted by Turkish morphology and silently discard good text:

1. **Stop-word presence.** Gopher rejects documents with too few stop words. That needs a
   *Turkish* stop-word list — an English list rejects legitimate Turkish prose wholesale.
2. **Mean word length.** Turkish is agglutinative (`ev → evler → evlerimizden →
   evlerimizdenmiş`), so the English 3.0–10.0 character band is too tight.

Rather than hand-tune these, we **derive** them from a clean Turkish reference corpus. This
is FineWeb-2's **"canary language"** approach, where Turkish was one of the tuned languages:
read the language's real parameters off a known-good corpus instead of guessing.

## The method

`scripts/calibrate_filters.py` streams **Turkish Wikipedia** (`wikimedia/wikipedia`,
pre-parsed clean prose) and feeds article texts into
`turkish_corpus.calibration.calibrate_from_corpus`, which:

1. **Stop-word list** — counts word frequencies (`text.split()`, each Turkish-lowercased via
   `turkish_lower` so the ı/İ distinction is preserved; only all-alphabetic tokens count) and
   takes the `--top-stopwords` most frequent words. These ARE the language's function words.
2. **Thresholds** — accumulates per-document Gopher metrics (`n_words`, `mean_word_length`,
   `symbol_word_ratio`, `non_alpha_words_ratio`) and reads the filter bands off their observed
   distribution via one of three strategies:
   - `quantile` (default): the `q` and `1 − q` percentiles (default `q = 0.005`).
   - `10tail`: the fixed 0.10 / 0.90 percentiles (a gentler tail trim).
   - `meanstd`: `mean ± 3·std`, clamped to the observed min/max.

   Floors are applied so a *clean* corpus does not produce near-zero caps that would later
   reject ordinary, slightly-noisier-but-fine web text: `max_symbol_word_ratio ≥ 0.05`,
   `max_non_alpha_words_ratio ≥ 0.2`, `min_doc_words ≥ 10`. **`max_doc_words` is kept
   generous** — floored at the `100_000` default (ceiled at `1_000_000`) and deliberately NOT
   pulled down to the reference's quantile: Wikipedia articles are short, but the production
   blend includes long-form registers (academic theses, laws) that legitimately run tens of
   thousands of words, so a Wikipedia-derived upper bound would silently drop entire theses.
   The line-based ratios (`max_bullet_lines_ratio`, `max_ellipsis_lines_ratio`) and
   `min_stop_words` are NOT calibrated here and keep their defaults.

Quantiles use a pure-Python linear-interpolation helper (`_quantile`) — no numpy dependency.

## Running it

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/berkayra/dev/turkish-corpus

# Default: 5000 articles from the 20231101.tr dump, quantile strategy.
uv run --extra sources python scripts/calibrate_filters.py --limit 5000

# More articles, explicit dump / strategy / tail fraction:
uv run --extra sources python scripts/calibrate_filters.py \
    --limit 20000 --dump 20231101.tr --strategy quantile --q 0.005
```

CLI flags: `--limit` (article sample size, default 5000), `--dump` (config id, default
`20231101.tr` — verify current configs on the dataset card before pinning), `--top-stopwords`
(default 175), `--strategy` (default `quantile`), `--q` (default 0.005), `--out-dir` (default
the package data dir). `datasets` is imported lazily (the `sources` extra).

## Where artifacts land

Into `src/turkish_corpus/data/` (resolved via `importlib.resources`):

| File | Contents |
|------|----------|
| `turkish_stopwords.txt` | One Turkish-lowercased stopword per line (UTF-8). |
| `turkish_thresholds.json` | The numeric Gopher/quality threshold kwargs (no `stop_words`). |
| `calibration_report.json` | Provenance: `n_docs`, vocabulary size, chosen values, raw metric quantiles. |

These are committed so the derived filters travel with the package and are diffable.

## How `filters.py` loads them (with curated fallback)

`turkish_corpus.filters` reads the artifacts transparently:

- `TURKISH_STOPWORDS = _load_stopwords()` reads `turkish_stopwords.txt` via
  `importlib.resources` when present and non-empty; otherwise it returns the curated
  `_CURATED_STOPWORDS` baked into the module.
- `TurkishQualityThresholds.calibrated()` loads `turkish_thresholds.json` and merges it over
  the dataclass defaults (with `stop_words = TURKISH_STOPWORDS`); when the file is absent it
  returns plain defaults.

So **before** you run the calibration the filters use the curated list + defaults; **after**,
they use the derived artifacts — with no code change and no breakage either way. The dataclass
validates its bands in `__post_init__`, so a malformed derived artifact fails fast rather than
silently rejecting an entire pipeline run.

The pipeline uses them by default: `PipelineConfig.quality` defaults to
`TurkishQualityThresholds.calibrated()`, so `tc-run-hplt` / `tc-build-corpus` pick up the
derived stopwords + thresholds automatically.

## Derived values (committed run: 5000 articles, `20231101.tr`, quantile q=0.005)

| Parameter | Hand-set default | **Derived** |
|-----------|-----------------:|------------:|
| stop words | 118 (curated) | **175** (frequency-ranked: `ve, bir, bu, olarak, ile, da, …`) |
| `min_doc_words` | 50 | **11** |
| `max_doc_words` | 100 000 | **100 000** (kept generous — see above) |
| `min_avg_word_length` | 3.0 | **4.96** |
| `max_avg_word_length` | 12.0 | **8.18** |
| `max_symbol_word_ratio` | 0.10 | **0.05** |
| `max_non_alpha_words_ratio` | 0.70 | **0.28** |

The headline result: `max_avg_word_length` came out **8.18**, *tighter* than the 12.0 guess —
per-document mean word length averages out, so the intuition that "Turkish needs a wide band"
over-corrected. Provenance is in `calibration_report.json` (vocab size 226 146 unique words).

**Register caveat:** these bands are from *encyclopedic* Wikipedia. The blend's other
registers (web, legal, academic) have somewhat different distributions; the `q=0.005` tails
are conservative (keep ~99% of the middle), but if you observe over-filtering of a specific
register, re-run on a register-mixed reference sample. `max_doc_words` is already protected.

## Re-running

Re-running `scripts/calibrate_filters.py` overwrites the artifacts in place (different
`--dump`, `--limit`, `--strategy`, or `--q` regenerate them). Commit the regenerated files to
update the derived filters.
