# The HPLT v2 Turkish cleaning pipeline

Anchored on **HPLT v2 cleaned `tur_Latn`** (~51.7B tokens, CC0). The flow mirrors
FineWeb-2's Turkish-validated recipe (Turkish was one of its tuned "canary" languages) but
swaps the data source for HPLT's CC0 license.

## Stages

```
Stage A — process            Stage B — minhash (near-dedup)      Stage C — finalize
─────────────────            ──────────────────────────────      ──────────────────
reader (HPLT hf | jsonl)     signatures  (per-doc MinHash)       reader (intermediate)
  ↓                            ↓                                    ↓
LanguageFilter (tr gate)     buckets     (group candidates)      MinhashDedupFilter
  ↓                            ↓            tasks = num_buckets     (drop duplicates,
GopherRepetitionFilter       cluster     (build remove-id sets)    keep one per cluster)
  ↓                            ↓            tasks = 1                ↓
GopherQualityFilter          → remove_ids/                        TokensCounter
  (Turkish thresholds                                               (your tokenizer)
   + stopwords)                                                      ↓
  ↓                                                                JsonlWriter → final/
FineWebQualityFilter
  ↓
TurkishNormalizer  (NFC, control/zero-width strip, preserve case)
  ↓
TurkishPIIRedactor (TC Kimlik / phone / IBAN)
  ↓
PIIFormatter       (emails / IPs)
  ↓
JsonlWriter → intermediate/
```

MinHash is split into stages because bucketing needs every document's signature; the three
sub-stages run as dependent executors. Output layout under `config.output_path`:

```
intermediate/        Stage A output (filtered, normalized, scrubbed)
minhash/signatures/  per-document signatures
minhash/buckets/     bucketed candidate pairs
minhash/remove_ids/  ids to drop
removed_duplicates/  audit trail of dropped near-duplicates
final/               the clean corpus
```

## Configuration

All knobs live in `turkish_corpus.config.PipelineConfig` (validated dataclasses). Key
defaults:

| Field | Default | Notes |
|-------|---------|-------|
| `reader.dataset` / `language` | `HPLT/HPLT2.0_cleaned` / `tur_Latn` | the CC0 anchor |
| `reader.source` | `hf` | `hf` streams from the Hub; `jsonl` reads downloaded shards |
| `quality.max_avg_word_length` | `12.0` | widened from English 10.0 for agglutination |
| `quality.stop_words` | `TURKISH_STOPWORDS` | replace with a frequency-derived list |
| `minhash` | 14 buckets × 8, 5-grams, 64-bit | FineWeb-2's Turkish-validated settings |
| `language.threshold` | `0.65` | secondary fastText `tr` gate (HPLT is pre-tagged) |
| `pii.*` | TC/phone/IBAN on, plate off | KVKK scrubbing |
| `tokenizer.name_or_path` | `None` | set to your `tokenizer.json` to count tokens |

## Running

```bash
uv sync --extra pipeline

# dev slice (streams a few thousand docs from the Hub)
uv run tc-run-hplt --limit 2000 --output ./output/dev --tasks 2

# production (Linux box) from pre-downloaded shards, with your tokenizer
uv run tc-run-hplt --source jsonl --data-path /data/hplt/tur_Latn \
    --tokenizer /models/tr-morph/tokenizer.json \
    --output /data/corpus/hplt_tur --tasks 64
```

For the full HPLT `tur_Latn` run, **download the shards once** (`source=jsonl`) rather than
re-streaming from the Hub. At cluster scale, mirror `run_local` with
`SlurmPipelineExecutor` (see `run_slurm` in `pipeline.py` — a documented seam to wire when
the Slurm box specs are known).

## Production checklist

- [ ] Download HPLT v2 `tur_Latn` shards to the Linux box; set `--source jsonl`.
- [ ] Derive a frequency-based Turkish stop-word list from Turkish Wikipedia; replace
      `TURKISH_STOPWORDS`.
- [ ] Calibrate Gopher/quality thresholds from a clean Turkish reference corpus
      (FineWeb-2 "10Tail"/"Quantile" strategy) rather than the hand-set defaults.
- [ ] Point `--tokenizer` at the morphology-aware `tokenizer.json` and record the final
      token count (with fertility vs. whitespace).
- [ ] Verify PII stats in the run logs; spot-check `removed_duplicates/` and `final/`.
- [ ] Keep a provenance record (source, date, license=CC0, intended use) for KVKK.
