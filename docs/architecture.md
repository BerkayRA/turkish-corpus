# Architecture

The package is split into a **pure core** (no heavy dependencies, fully unit-tested on any
machine) and a **datatrove integration layer** (imported on demand). This is deliberate:
the Turkish-specific logic is the part that carries risk and needs tight tests, and it
should not require a multi-hundred-MB install to develop or verify.

```
                         turkish_corpus
                         ──────────────
  PURE CORE (deps: tokenizers only)        DATATROVE LAYER (extra: pipeline)
  ─────────────────────────────────        ────────────────────────────────
  normalization.py   ı/İ casing, NFC,       blocks.py     TurkishNormalizer,
                     control/zero-width                    TurkishPIIRedactor
                     cleanup                               (wrap pure core as
  pii.py             TC Kimlik / phone /                   datatrove PipelineSteps)
                     IBAN recognizers        pipeline.py   build_process_pipeline(),
  tokenizer.py       pluggable token                       run_local(), run_slurm()
                     counter (your tok.)                   — assembles reader → filters
  filters.py         Turkish stopwords +                   → normalize → PII → minhash
                     Gopher thresholds                     → token count → write
  config.py          dataclasses + validate  cli.py        tc-run-hplt entry point
        │                                          │
        └───────────────── consumed by ───────────┘
```

## Dependency boundary

- `import turkish_corpus` pulls in **only** the pure core (`tokenizers` is the single
  runtime dep, used lazily by `HFTokenCounter`).
- `turkish_corpus.blocks` and `turkish_corpus.pipeline` import `datatrove` at call time
  and raise a clear, actionable error if the `pipeline` extra is missing.
- Optional capabilities degrade gracefully: PyICU casing (`--extra icu`) and ftfy
  encoding repair (`--extra encoding`) fall back to correct pure-Python behavior when
  absent.

## Why the pure/impure split pays off

1. **Tests run in 0.1s** on a clean checkout without datatrove, so the Turkish casing and
   PII logic — the bug-prone part — is always under CI-speed verification.
2. **The same cleaning core serves both tools.** The future Scrapy crawler (roadmap step
   3) will feed crawled WARC/HTML through the *identical* `blocks` + `pipeline`, so there
   is one cleaning implementation, not two (DRY).
3. **Portability.** Dev on macOS, production on the Linux box: only the executor changes
   (`LocalPipelineExecutor` → `SlurmPipelineExecutor`); the pipeline definition is shared.

## Data flow contract

datatrove passes `Document` objects (`.text`, `.id`, `.metadata`) through a generator of
`PipelineStep`s. Our custom steps mutate `doc.text` in place per datatrove convention and
emit per-type `stat_update` counters (e.g. `pii_tc_kimlik`) so every run produces an
auditable record of what was filtered and redacted — important for KVKK provenance.
