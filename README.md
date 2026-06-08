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

**Whole corpus in one command** — ingest → clean → blend from a recipe (see
[`docs/end-to-end.md`](docs/end-to-end.md)):

```bash
uv run --extra pipeline --extra sources --extra academic \
    tc-build-corpus --recipe examples/recipe.json
# → output/corpus/{raw,clean,blend}/ + corpus_manifest.json (per-source token accounting)
```

Or drive the stages individually:

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

## Sibling repos (tokenizers)

This corpus is tokenized with the user's own tokenizers, wired in via
[`docs/tokenizer.md`](docs/tokenizer.md):

- [`BerkayRA/turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer) — pure-Python
  morphological **segmenter** (`tr_api`), used here for *fertility analysis*.
- [`BerkayRA/turkish-llm`](https://github.com/BerkayRA/turkish-llm) — where the
  **morpheme-aware BPE** (and SentencePiece/byte-BPE baselines) are trained and exported.
  Export it to HF `tokenizer.json` and pass `--tokenizer` to count real corpus tokens.

## Architecture

Pure, dependency-free Turkish logic (`normalization`, `pii`, `tokenizer`, `filters`,
`config`, `fertility`, `crawl.cc_index`, `crawl.seeds`) is unit-tested on any machine; the
heavy integrations (`blocks`, `pipeline`, `morphology`, `crawl.spider`/`pipelines`) are
imported on demand and exercised under the `pipeline`/`sentencepiece`/`crawl` extras. Full
diagram in [`docs/architecture.md`](docs/architecture.md); the one-command orchestrator in
[`docs/end-to-end.md`](docs/end-to-end.md); pipeline stages in
[`docs/pipeline.md`](docs/pipeline.md); tokenizers in [`docs/tokenizer.md`](docs/tokenizer.md);
sources in [`docs/sources.md`](docs/sources.md); the blend in [`docs/blend.md`](docs/blend.md);
data acquisition (downloaders/scrapers) in [`docs/acquisition.md`](docs/acquisition.md);
crawler in [`docs/crawler.md`](docs/crawler.md).

## Roadmap

Anchored on the research recommendation — see [`docs/roadmap.md`](docs/roadmap.md):

1. ✅ **Pipeline foundation** — Turkish cleaning pipeline, verified end-to-end on real HPLT.
2. ✅ **Register-diverse blend** — Wikipedia + government/legal + academic (YÖKTEZ/DergiPark)
   ingesters + a token-budget [mixer](docs/blend.md) (manifest, sized ~15–25B for a competent
   model). See [`docs/sources.md`](docs/sources.md).
3. ✅ **Crawler** — Common Crawl index seed query (DuckDB) + Scrapy focused crawler, sharing
   this datatrove cleaning backend (output feeds `tc-run-hplt --source jsonl`).
4. **Train + measure** — tokenize with the morpheme-aware BPE; report tokens & fertility.

## Development

```bash
uv run pytest --cov=src/turkish_corpus --cov-report=term-missing
uv run ruff check src tests
```

The datatrove-dependent code is covered by tests marked `pipeline` (run after
`uv sync --extra pipeline`). License: Apache-2.0 (code). Corpus license follows the source
data — HPLT v2 is CC0.
