# Blending the corpus to a token budget

`turkish_corpus.blend` mixes the **cleaned** source dirs into the final register-diverse
corpus at a target token count, measured with the configured tokenizer, and writes an
auditable `manifest.json`.

## Sizing — how big should the corpus be?

The target is **not** a hard 1B tokens. For a *competent* Turkish LLM, size by the model you
intend to train (Chinchilla-optimal ≈ 20 tokens/parameter):

| Model scale | Chinchilla-optimal tokens | Notes |
|-------------|---------------------------|-------|
| ~350M params | ~7B | small but usable |
| ~1B params | ~20B | solid small LLM |
| ~1.5–3B params | ~30–60B | competitive Turkish LLM |

**Recommended target: ~15–25B high-quality tokens** for a ~1B-class model, scalable upward
— HPLT `tur_Latn` alone provides ~51.7B, so the web anchor can fill any budget while the
curated registers (Wikipedia, legal, academic) supply diversity. Pass `--target-tokens`
to set it; count with the real tokenizer (`--tokenizer`) so the budget is in *your* model's
tokens, not whitespace words.

## Running the blend

```bash
uv run python scripts/build_blend.py \
    --source hplt=output/clean/hplt_tur/final:0.60 \
    --source wikipedia=output/clean/wikipedia/final:0.12 \
    --source mevzuat=output/clean/govlegal/final:0.10 \
    --source academic=output/clean/academic/final:0.18 \
    --target-tokens 20e9 \
    --tokenizer /models/turkish-llm/morpheme_bpe.json \
    --out output/blend
```

`--target-tokens` accepts `20e9`, `20_000_000_000`, or a plain int. Weights are normalized to
sum 1.0. Selection is a deterministic shuffle (`--seed`) so re-runs are reproducible and a
source isn't biased to its first documents; achieved tokens may overshoot a per-source target
by at most one document (reported exactly). If a source has fewer tokens than its target, the
`shortfall` is recorded rather than hidden.

### A starting recipe (~20B, scale weights to taste)

| Source | Weight | Register | Why |
|--------|-------:|----------|-----|
| HPLT web | 60% | web | broad fluent coverage; the scalable bulk (CC0) |
| Academic (YÖKTEZ/DergiPark) | 18% | academic | highest-quality formal long-form |
| Wikipedia | 12% | encyclopedic | cleanest; encyclopedic anchor |
| Legal (mevzuat/…) | 10% | legal | formal register; cap to avoid legalese skew |

Add news / OpenSubtitles later for journalistic / conversational registers.

## The manifest

`out_dir/manifest.json` records, per source: `weight`, `target_tokens`, `achieved_tokens`,
`available_tokens`, `docs`, `shortfall_tokens`, plus `license`/`register`; and `totals`
(target/achieved/docs/tokenizer). It's the reproducibility + licensing audit for the blend.

## Cross-source dedup (optional)

Per-source near-dedup already happens in the cleaning pipeline. Web sources overlap with
news/wiki, so an optional `cross_source_dedup(...)` runs the same 4-stage MinHash topology
(signatures → buckets → cluster → filter) over the **union** of source dirs before blending
(needs `--extra pipeline`). For a handful of distinct registers the overlap is small; run it
if you add multiple web-derived sources.
