# turkish-corpus

Pipeline (and, soon, crawler) for assembling a **high-quality ~1B-token Turkish (Türkçe)
LLM pretraining corpus**.

The data problem at the 1B-token scale is **curation, not scarcity**: 60–70B+ tokens of
already-cleaned, openly-licensed Turkish text exist. This repo anchors on **HPLT v2
cleaned `tur_Latn`** (~51.7B tokens, **CC0 license** — chosen over FineWeb-2 for the
cleanest licensing) and cleans it into a high-quality corpus with a Turkish-aware
[datatrove](https://github.com/huggingface/datatrove) pipeline. The full background
research lives in [`docs/research-summary.md`](docs/research-summary.md).

## Why this isn't just `datatrove fineweb.py`

Generic pipelines silently corrupt Turkish. This package fixes the parts that matter:

- **The `ı/İ` casing bug.** Python's `str.lower()` maps `I → i` (wrong: Turkish needs
  `ı`) and `"İ".lower()` returns two characters. Lowercasing before dedup hashing makes
  `"KISA"` (→ `"kısa"`) collide with the unrelated `"kisa"`, wrecking MinHash precision.
  We provide correct `tr`-locale casing (pure-Python, with optional PyICU) that preserves
  the `ı`/`i` distinction.
- **Agglutination-aware filters.** Turkish word lengths are longer; the Gopher mean-word-
  length band and stop-word checks are retuned, with a Turkish stop-word list.
- **KVKK PII scrubbing.** Custom recognizers for **T.C. Kimlik No** (checksum-validated),
  Turkish phone numbers, and **TR IBAN** (mod-97 validated), on top of datatrove's
  email/IP redaction.
- **Pluggable token counting** for your **morphology-aware tokenizer** — the token budget
  must be measured with the real tokenizer, not a generic over-fragmenting one.

See [`docs/turkish-gotchas.md`](docs/turkish-gotchas.md) for the full list with examples.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                       # core + dev (pure Turkish modules, tests)
uv sync --extra pipeline      # + datatrove (run the cleaning pipeline)
uv sync --extra icu           # + PyICU tr-locale casing (needs: brew install icu4c)
uv sync --extra encoding      # + ftfy mojibake repair (for fresh crawl text)
uv sync --extra crawl         # + duckdb/trafilatura (crawler tool, roadmap step 4)
```

## Quickstart

```bash
# Validate the pipeline config without running (no heavy deps needed):
uv run tc-run-hplt --limit 2000 --output ./output/dev --dry-run

# Small local dev slice (streams from the Hub; needs the pipeline extra):
uv sync --extra pipeline
uv run tc-run-hplt --limit 2000 --output ./output/dev --tasks 2

# Full run from pre-downloaded JSONL shards on the Linux box, counting tokens with
# YOUR morphology-aware tokenizer:
uv run tc-run-hplt \
    --source jsonl --data-path /data/hplt/tur_Latn \
    --tokenizer /models/tr-morph/tokenizer.json \
    --output /data/corpus/hplt_tur --tasks 64
```

Use the pure helpers directly:

```python
from turkish_corpus import turkish_lower, redact_turkish_pii, load_token_counter

turkish_lower("DİYARBAKIR KISA")          # -> "diyarbakır kısa"
redact_turkish_pii("IBAN: TR33...").text  # -> "IBAN: <IBAN>"
load_token_counter("/models/tr-morph/tokenizer.json").count("evlerimizden")
```

## Architecture

Pure, dependency-free Turkish logic (`normalization`, `pii`, `tokenizer`, `filters`,
`config`) is unit-tested on any machine; the datatrove integration (`blocks`, `pipeline`)
is imported on demand and exercised when the `pipeline` extra is installed. Full diagram
in [`docs/architecture.md`](docs/architecture.md); pipeline stages in
[`docs/pipeline.md`](docs/pipeline.md).

## Roadmap

Anchored on the research recommendation — see [`docs/roadmap.md`](docs/roadmap.md):

1. ✅ **Pipeline foundation** — Turkish cleaning pipeline config (this repo).
2. Register-diverse blend (Wikipedia, news, OpenSubtitles, OCR'd YÖKTEZ theses).
3. **Crawler** — Common Crawl index seed query (DuckDB) + Scrapy focused crawler, sharing
   this datatrove cleaning backend.

## Development

```bash
uv run pytest --cov=src/turkish_corpus --cov-report=term-missing
uv run ruff check src tests
```

The datatrove-dependent code is covered by tests marked `pipeline` (run after
`uv sync --extra pipeline`). License: Apache-2.0 (code). Corpus license follows the source
data — HPLT v2 is CC0.
