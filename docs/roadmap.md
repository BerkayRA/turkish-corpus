# Roadmap

From the research recommendation (full report in
[`research-summary.md`](research-summary.md)). The corpus target is ~1B high-quality
tokens; the binding constraint is curation quality, not data availability.

## Step 1 — Pipeline foundation ✅ (this repo)

Turkish-aware datatrove cleaning pipeline anchored on **HPLT v2 `tur_Latn` (CC0)**:
normalization (`ı/İ` casing), agglutination-tuned filters, KVKK PII scrubbing, MinHash
near-dedup (Turkish word tokenizer), pluggable token counting. Pure core fully unit-tested.
**Verified end-to-end** on a real 300-doc HPLT slice (300 → 190 after filters → 1 near-dup
dropped → 189 clean docs).

**Tokenizer wiring ✅** (see [`tokenizer.md`](tokenizer.md)): `SentencePieceTokenCounter` +
HF `tokenizer.json` support for the [`turkish-llm`](https://github.com/BerkayRA/turkish-llm)
exports; a `morphology` bridge to [`turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer)
(`tr_api`) and a fertility tool (`scripts/measure_fertility.py`). When the morpheme-aware BPE
is trained in `turkish-llm`, export it to HF `tokenizer.json` and pass `--tokenizer`.

**Remaining within this step:** derive a frequency-based stop-word list and calibrate
quality thresholds from Turkish Wikipedia (currently sensible hand-set defaults).

## Step 2 — Register-diverse blend ✅ (in progress → core built)

Built the source framework + ingesters + token-budget mixer (see [`sources.md`](sources.md)
and [`blend.md`](blend.md)). **Target is no longer a hard 1B** — size by model scale
(~15–25B for a ~1B-param model; HPLT alone has 51.7B to fill the budget). Ingesters:
Wikipedia ✅, government/legal (mevzuat ✅; Resmî Gazete/courts/TBMM scaffolded), academic
(DergiPark/YÖKTEZ PDF→text ✅; downloaders + OCR-for-scans scaffolded). Each is cleaned by the
existing pipeline (`tc-run-hplt --source jsonl`) then combined by `build_blend` with weights,
a token budget, and a manifest.

**Remaining within this step:** write the gov/legal scrapers and the DergiPark/YÖKTEZ
downloaders (+ optional OCR for scanned theses); run the real ingest→clean→blend at scale.

Original recipe sketch (web ↔ news ↔ wiki overlap is real, so dedup across sources):

| Domain | Target share | Source | Effort |
|--------|-------------:|--------|--------|
| Curated web | 35% | HPLT v2 (this pipeline) | done |
| Academic | 18% | YÖKTEZ theses + DergiPark | PDF download + OCR |
| News | 15% | TS Timeline + BOUN corpora | download |
| Wikipedia | 12% | trwiki dumps | download + WikiExtractor |
| Gov/legal (capped) | 8% | mevzuat, courts, TBMM | scrape / HF mirror |
| Subtitles (capped) | 7% | OpenSubtitles (OPUS) | download |
| Books | 3% | Wikisource / public domain | download |
| Q&A | 2% | Stack Exchange (CC BY-SA only) | dump |

All routed through the **same** `blocks` + `pipeline` cleaning backend, then a final
cross-source MinHash dedup pass (collect ~1.15B raw → 1.0B clean). The main custom-
collection work is OCR'ing YÖKTEZ theses.

## Step 3 — Crawler (Common Crawl + Scrapy) ✅

Built (see [`crawler.md`](crawler.md)). A crawler is **optional** at 1B tokens but available
as a targeted top-up.

- **3a — CC index seed query (`--extra crawl`).** DuckDB over Common Crawl's columnar
  Parquet index: filter `content_languages = tur`, group by host, rank by Turkish page
  count → an empirical, traffic-weighted `.tr` seed list. Optionally fetch only needed
  WARC byte ranges (UnifiedCrawl method: filter all of CC for <$4/<1 day).
- **3b — Scrapy focused crawler.** Frontier (front queues + per-host back queues), Bloom
  seen-set, robots.txt cache + per-host token-bucket politeness, trafilatura extraction,
  WARC output → **same datatrove cleaning backend** as steps 1–2. KVKK: aggressive PII
  scrub, avoid PII-dense forums, provenance log.

Verdict from research: **mine Common Crawl before crawling fresh** (cheaper, faster, lower
legal risk); the custom crawler only adds high-value `.tr` content CC under-covers.

## Step 4 — Train + measure

Tokenize the final corpus with the morphology-aware tokenizer; report token count and
fertility; validate the blend on a small training run before scaling.
