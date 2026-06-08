# Roadmap

From the research recommendation (full report in
[`research-summary.md`](research-summary.md)). The corpus target is ~1B high-quality
tokens; the binding constraint is curation quality, not data availability.

## Step 1 — Pipeline foundation ✅ (this repo)

Turkish-aware datatrove cleaning pipeline anchored on **HPLT v2 `tur_Latn` (CC0)**:
normalization (`ı/İ` casing), agglutination-tuned filters, KVKK PII scrubbing, MinHash
near-dedup, pluggable token counting. Pure core fully unit-tested; pipeline assembled and
config-validated.

**Next within this step:** run a dev slice with the `pipeline` extra; wire the
morphology-aware tokenizer; derive a frequency-based stop-word list and calibrate
thresholds from Turkish Wikipedia.

## Step 2 — Register-diverse blend

A web-only corpus is fluent but narrow. Mix in higher-quality registers, deduped across
sources (web ↔ news ↔ wiki overlap is real):

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

## Step 3 — Crawler (Common Crawl + Scrapy)

A crawler is **optional** at 1B tokens but planned as a targeted top-up.

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
