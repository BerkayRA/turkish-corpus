# Building a ~1B-Token High-Quality Turkish (Türkçe) LLM Corpus
*Generated: 2026-06-08 | Sources: ~70 across 4 research streams | Confidence: High*

## Executive Summary

The premise of the task — "I am lacking high-quality, high-volume data" — turns out to be **mostly a solved problem at the 1B-token scale**. Turkish is a *high-resource* language inside every major web corpus: it is ~1.0–1.3% of Common Crawl, and ready-cleaned, openly-licensed Turkish datasets already exceed **60–70 billion tokens** ([CulturaX `tr` = 64.3B tokens](https://huggingface.co/datasets/uonlp/CulturaX); [HPLT v2 cleaned `tur_Latn` = 51.7B tokens, CC0](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned); [FineWeb-2 `tur_Latn` = 41.9B words](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)). Your 1B-token goal is **50–65× smaller than a single one of these sources.**

Therefore the real engineering problem is **(a) which clean source to anchor on, (b) how to blend registers for quality, (c) a Turkish-aware cleaning/dedup pipeline, and (d) optionally a focused crawler for fresh/high-value content CC misses.** A web crawler is **not required** to hit 1B tokens — but it is worth building as a *targeted supplement*, and a concrete plan is included.

**Recommendation in one line:** Anchor on **FineWeb-2 `tur_Latn`** (or HPLT v2 for CC0 licensing), blend in Turkish Wikipedia + academic theses + news, run a **datatrove / FineWeb-2 Turkish-tuned** cleaning pipeline, and build a small **Scrapy crawler seeded empirically from the Common Crawl index** only as a top-up. Total compute to assemble 1B clean tokens: a single workstation, hours-to-days, ~$50–500.

---

## 1. Options ranked by Complexity-to-Build vs. Availability

| # | Option | Tokens available (Turkish) | Availability | Complexity to build | Quality | License |
|---|--------|---------------------------|-------------|---------------------|---------|---------|
| 1 | **FineWeb-2 `tur_Latn`** (download) | ~42B words (~55–65B BPE) | ⭐ Ungated HF, stream per-lang | **Trivial** — `load_dataset(..., "tur_Latn", streaming=True)` | Highest of web sets (per-lang filters, MinHash, ftfy encoding fix) | ODC-BY + CC ToU |
| 2 | **HPLT v2 cleaned `tur_Latn`** (download) | 51.7B tokens | ⭐ Ungated HF | **Trivial** | High (deduped + heuristic clean) | **CC0** (best for commercial) |
| 3 | **CulturaX `tr`** (download) | 64.3B tokens | Gated HF (accept terms) | **Trivial** (after gate) | High (mC4+OSCAR cleaned, MinHashLSH) | Inherited mC4/OSCAR — verify for commercial |
| 4 | **Turkish Wikipedia** (download) | ~0.4–0.6B BPE | ⭐ Direct dumps | **Low** (WikiExtractor) | **Cleanest source** — encyclopedic | CC BY-SA (share-alike) |
| 5 | **Academic: YÖKTEZ theses + DergiPark** | theses ~9.6B; DergiPark several B | Portals; PDF download | **Medium** (PDF→text / OCR) | **Highest** long-form formal | Theses author-permitted; DergiPark mostly CC BY |
| 6 | **News corpora (TS Timeline, BOUN)** | >1B (TS=700M+ words, BOUN~500M) | Research download (some registration) | **Low** | High journalistic | Murky (research-use) |
| 7 | **Gov/legal (courts, mevzuat, Resmî Gazete, TBMM)** | >5B (Yargıtay alone ~3.4B) | Portals + 1 HF mirror | **Medium** (scraping) | High but narrow/repetitive | **Best legal status** (public official docs) |
| 8 | **OpenSubtitles / OPUS `tr`** | ~1B words | ⭐ Direct | **Trivial** | Medium (conversational, fragmented) | Gray (research-standard) |
| 9 | **Mine Common Crawl yourself** | tens of B per snapshot | Free (AWS-hosted) | **Medium** (datatrove/UnifiedCrawl) | You control it (FineWeb-grade) | CC ToU |
| 10 | **Custom focused crawler** | unlimited (fresh) | You build it | **High** (4–8 wk eng) | You control it; freshest | You own the legal exposure |
| 11 | **Forums/social (Ekşi, Reddit)** | 1B+ (Ekşi) but… | Scraping | **High + legally hostile** | Diverse colloquial, noisy | **Worst** (ToS-prohibited) — avoid |

**Reading the table:** Options 1–4 and 8 are essentially free downloads that *already exceed your target*. Options 5–7 add register diversity and the best licensing, at the cost of PDF/scrape work. Options 9–10 are only justified for freshness or domains CC misses. Option 11 should be excluded for a clean, redistributable corpus.

---

## 2. Recommended 1B-Token Recipe (least effort, high quality)

Two valid blends depending on how much custom collection you want to do.

### Blend A — "Download-only" (fastest; ~1 week, near-zero custom work)
- **85%** FineWeb-2 `tur_Latn` (quality-filtered slice)
- **10%** Turkish Wikipedia
- **5%** native instruction/curated text (e.g. InstrucTurca)
- → Clean, balanced 1B mix with essentially no custom cleaning. Pick **one** web source to avoid cross-source dedup.

### Blend B — "Register-diverse" (best model quality; ~3–4 weeks)
Token counts are subword (LLM) tokens; assumes aggressive cross-source MinHash dedup (collect ~1.15B raw → 1.0B clean).

| Domain | Share | Tokens | Source |
|--------|------:|-------:|--------|
| Curated web (FineWeb-2/CulturaX top slice) | 35% | 350M | web |
| Academic (YÖKTEZ theses + DergiPark) | 18% | 180M | PDF/OCR |
| News (TS Timeline + BOUN, deduped) | 15% | 150M | research corpora |
| Wikipedia + Wikisource | 12% | 120M | dumps |
| Gov/legal (courts + mevzuat + TBMM) — **capped** | 8% | 80M | portals |
| Subtitles/dialogue (OpenSubtitles) — **capped** | 7% | 70M | OPUS |
| Books/literature (PD + Wikisource) | 3% | 30M | Gutenberg/Wikisource |
| Forums (Stack Exchange CC BY-SA only) | 2% | 20M | SE dumps |
| **Total** | **100%** | **~1.0B** | |

Caps on legal/subtitles matter: they are internally repetitive (boilerplate, repeated subtitle lines) and register-narrow — over-weighting teaches legalese/fragmented dialogue. ~60% of Blend B is highest/high tier (academic + wiki + news + books); only ~9% is medium register-specific.

**Note on "tokens":** corpus tables usually report *words*, not subword tokens. Turkish is agglutinative, so a BPE tokenizer yields ~1.3–1.6 subword tokens/word — and multilingual tokenizers over-fragment Turkish badly (a "tokenization premium" up to 10–15× vs. English). Since **you already have a morphology-aware tokenizer**, your real token counts will be substantially better than a multilingual baseline — count tokens with *your* tokenizer, not a generic one.

---

## 3. Cleaning, Filtering & Deduplication Pipeline

Canonical order: **WARC → extract → URL filter → language ID → heuristic quality → dedup → (model quality) → PII/toxicity → tokenize.** If you start from a pre-extracted source (FineWeb-2/CulturaX/HPLT), skip the first two stages.

**Tooling per stage (open-source):**
- **HTML extraction:** trafilatura (primary, FineWeb's choice) + resiliparse (fast, better tables). Union of extractors raised one project's yield 71% — worth it given Turkish volume.
- **Language ID:** GlotLID (script-aware, FineWeb-2's choice) > fastText lid176 > avoid CLD3 (mC4 was criticized for it).
- **Heuristic filters:** Gopher + C4/FineWeb rules — but **re-derive thresholds from a Turkish reference corpus** (FineWeb-2 "10Tail"/"Quantile"/"MeanStd" strategies).
- **Model-based quality:** no off-the-shelf Turkish edu-classifier exists. Cheapest: KenLM perplexity trained on Turkish Wikipedia (CCNet style). Best: port the FineWeb-Edu recipe — score a few thousand Turkish docs 0–5 with a strong LLM, train a head on a Turkish embedding model (BERTurk).
- **Dedup:** layered — exact/line (CCNet SHA), MinHash+LSH (FineWeb-2: 14 buckets × 8, 5-grams, word-level, **global per language**), suffix-array substring. Tools: **datatrove**, text-dedup, NeMo Curator (GPU). FineWeb-2's "rehydration" (keep one per cluster but upsample by cluster size) is valuable for scarce Turkish.
- **PII (KVKK!):** Presidio + **custom Turkish recognizers** — TC Kimlik (11-digit, validatable checksum), +90/05xx phones, TR IBAN, plates.
- **Toxicity/NSFW:** Celadon + Turkish bad-word lexicon + UT1 URL blocklist (don't translate English lists).

### Turkish-specific gotchas your tokenizer/pipeline must handle
These are the silent corruption bugs generic pipelines get wrong — directly relevant to you:

1. **The dotless/dotted-i casing bug (#1 silent bug).** Turkish has `I↔ı` and `İ↔i`. Default `str.lower()` maps `I→i` (wrong; should be `ı`) and `"İ".lower()` even returns a 2-char string. This silently breaks **dedup hashes** (false dup/non-dup), **stop-word/bad-word matching**, and **filter ratios**. Fix: case with **PyICU `tr` locale** or `unicode_tr`; prefer NFC + ICU `tr` casefold for dedup keys (or don't lowercase in the hash at all).
2. **Diacritics & deasciification.** Lots of older/informal Turkish is ASCII ("Turkce", "ogrenci"). It lowers LID confidence (risk of wrongful discard). Detect+route it; optionally restore with `turkish-deasciifier` (selectively, high-confidence only — restoring beats stripping for downstream). Always NFC-normalize first so `ş/ğ` are single codepoints (stabilizes dedup).
3. **Morphology vs. dedup/filters.** Agglutination means inflected variants look distinct to word-n-gram MinHash (under-detects paraphrase dups), and shifts mean-word-length / alphabetic-word ratios — re-derive thresholds. For quality assessment, suffix entropy & lemma diversity beat raw type-token ratio.
4. **Stop-word list:** derive automatically from Turkish word-frequency (FineWeb-2 method) rather than using an English list.

**Best framework:** **datatrove with the FineWeb-2 `tur_Latn` recipe** — Turkish was an explicit "canary" language, so per-language thresholds, stop-word derivation, MinHash config, and rehydration are already implemented. Use NeMo Curator only if you have GPUs and scale past tens of billions.

**Compute for ~1B tokens:** single workstation. From a pre-extracted source (~15–50 GB to net 1B clean): low hundreds of CPU-core-hours ≈ a few hours–1 day on 32–64 vCPU, ~tens of dollars. Optional model-quality classifier adds a few GPU-hours + ~$tens–hundreds for LLM annotations. Total ~$50–500. The cost is *engineering the Turkish pieces*, not the compute.

---

## 4. Crawler Build Plan (if you choose to build one)

**Verdict first:** For ~1B tokens, **mine Common Crawl (Path A) before building a fresh crawler (Path B).** CC is free, AWS-hosted, ~10× cheaper/faster, lower legal risk, and Turkish is abundant (one snapshot >> 1B tokens). Build a custom crawler only as a *targeted top-up* for JS-heavy news SPAs, specific forums, or fresh post-2024 content CC under-covers.

### Path A — Common Crawl mining (recommended first)
The key cost-saver is the **columnar (Parquet) URL index** with `content_languages`: query it in DuckDB for `tur` rows, get exact WARC byte ranges, then HTTP **Range-request only those bytes** (never download full WARCs). Per UnifiedCrawl, the *entire* CC can be index-filtered for **<$4 in <1 day** on one server. For Turkish you only need 1–3 recent snapshots.
- Tools: **datatrove** (FineWeb-2-grade, replicable) or **UnifiedCrawl** (index+range, consumer hardware); cc_net / ungoliant (OSCAR) as alternatives.
- Effort: 1 engineer, ~1–2 weeks; time-to-1B: days; infra: single box.

### Path B — Custom focused crawler (the supplement)
Single-box, WARC-producing, polite, **sharing ~90% of its cleaning code with Path A** (DRY — same datatrove backend):

```
SEEDS                         URL FRONTIER                    FETCH            RAW            CLEAN (datatrove blocks)
.tr domain lists  ─┐    front queues (priority)            Scrapy (static)   WARC    WarcReader → Trafilatura
CC index top       ├──▶ selector → back queues (1/host)  ──▶ or Crawlee/    ──▶ files ─▶ → GlotLID(tur, per-lang thr.)
  Turkish hosts    │    URL norm + Bloom seen-set            Playwright              → Gopher/FineWeb(TR-tuned)
tr.wiki extlinks  ─┘    robots.txt cache + crawl-delay       (JS-heavy)              → MinHash + substring dedup
                        per-host token bucket + backoff                              → PII scrub (TR phone/TC Kimlik)
                                                                                     → TokensCounter → JSONL/Parquet
                                                                                     + provenance log
```

**Component choices & why:**
- **Seeds — seed empirically, not by guessing:** query the CC index (`content_languages LIKE '%tur%'`, GROUP BY host, ORDER BY count) to get traffic-weighted highest-yield Turkish hosts. Augment with [agmmnn/tr-domains](https://github.com/agmmnn/tr-domains) (~200k live .tr), zonefiles.io .tr, and `tr.wikipedia.org` external links. Useful subdomains: `edu.tr`, `gov.tr`/`bel.tr`, news on `.com.tr`.
- **Fetcher:** Scrapy default (mature, focused crawls); Crawlee/Playwright only per-domain for JS-rendered sites (cost control). Colly (Go) if you need raw throughput.
- **Frontier:** front queues by priority → selector → one back queue per host (politeness); URL normalization + Bloom filter seen-set; robots.txt fetched/cached *before* enqueue.
- **Politeness:** per-host token bucket, adaptive backoff on 429/5xx, honor Crawl-delay, honest User-Agent + contact URL, crawl off-peak.
- **Storage:** WARC (archival, CC-tooling-compatible) → JSONL/Parquet after cleaning.
- **Same cleaning backend as Path A** so you maintain one pipeline.

### Legal & ethical (Turkey-specific)
- **KVKK (Law 6698)** — GDPR-aligned, extraterritorial, automated collection = "processing"; penalties to ~TRY 13.6M, criminal exposure for unlawful PII recording. **Mitigation: aggressively PII-scrub**, avoid profile/forum pages dense with personal data, prefer editorial/encyclopedic content.
- **robots.txt + TDM opt-out** — always honor; EU framework (DSM Directive, AI Act) is the sensible baseline; 2024 German LAION ruling permitted scraping for *non-commercial research* training under the TDM exception.
- **Copyright/ToS** — no Turkish fair-use; rely on minimal retention, non-reproduction, research framing; don't bypass paywalls/logins; don't redistribute verbatim copyrighted text at scale.
- **Provenance log** — record source, date, robots/license state, intended use (increasingly a legal expectation).
- **Path A legal advantage:** Common Crawl did the crawling, so you process already-public data — materially lower direct exposure (copyright/KVKK on the *resulting corpus* still apply).

---

## 5. Recommended Roadmap

1. **Week 0 (proof):** Stream FineWeb-2 `tur_Latn`, tokenize ~1B with **your** tokenizer, sanity-check quality and your tokenizer's fertility on real data. You have a working corpus immediately (Blend A).
2. **Weeks 1–2 (pipeline):** Stand up datatrove with the FineWeb-2 Turkish recipe; add the Turkish-specific fixes (ICU `tr` casing, NFC, Turkish stop/bad-word/PII recognizers). Re-derive filter thresholds from Turkish Wikipedia.
3. **Weeks 2–4 (register diversity → Blend B):** Add Wikipedia, OpenSubtitles, news corpora (downloads); OCR a YÖKTEZ thesis slice + pull DergiPark/mevzuat. Cross-source MinHash dedup. This is the quality jump.
4. **Optional (weeks 4+):** Mine 1–3 CC snapshots (Path A) for more/fresher web; bolt on the Scrapy crawler (Path B) for specific high-value .tr domains the corpora miss — reusing the same cleaning backend.
5. **Licensing decision:** if commercial, prefer **HPLT v2 (CC0)** as the web anchor and avoid CulturaX's inherited-license ambiguity; keep Wikipedia's share-alike in mind.

---

## Key Takeaways

- **Sourcing is not your bottleneck.** 60–70B+ clean Turkish tokens are a free download; 1B is 50–65× over-covered. Reframe the project as curation + cleaning, not data hunting.
- **Fastest path to 1B:** FineWeb-2 `tur_Latn` (or HPLT v2 for CC0) + a 10% Wikipedia blend — ~1 week, minimal custom work.
- **Best-quality path:** register-diverse Blend B (web + theses + news + wiki + capped legal/subtitles), deduped across sources — ~3–4 weeks; the academic/thesis layer is the highest-value custom work.
- **A web crawler is optional.** If you build one, **mine Common Crawl first** (index + HTTP-range, <$4/<1 day to filter); build a Scrapy crawler only as a targeted supplement, seeded empirically from the CC host ranking, sharing one datatrove cleaning backend.
- **Turkish-specific cleaning is where projects silently fail:** fix the `ı/İ` casing bug (use ICU `tr`), NFC-normalize, handle deasciification, re-derive filter thresholds for agglutinative morphology, and add Turkish PII recognizers for KVKK. Your morphology-aware tokenizer is an asset here — count tokens with it, not a multilingual baseline.

---

## Sources (selected)

**Existing corpora:** [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) · [HPLT v2 cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) · [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) ([paper](https://arxiv.org/abs/2309.09400)) · [MADLAD-400](https://arxiv.org/pdf/2309.04662) · [BERTurk](https://github.com/stefan-it/turkish-bert) · [TS Corpus](https://tscorpus.com/corpora) · [cosmosGPT](https://arxiv.org/html/2404.17336v1) · [Resources for Turkish NLP survey](https://arxiv.org/pdf/2204.05042) · [awesome-turkish-language-models](https://github.com/kesimeg/awesome-turkish-language-models)

**Sources to 1B:** [Mecellem Turkish legal LLM (YÖKTEZ 9.6B, Yargıtay 3.4B tokens)](https://arxiv.org/html/2601.16018v1) · [OpenSubtitles2018](https://aclanthology.org/L18-1275.pdf) · [trwiki dumps](https://download.wikimedia.org/trwiki/) · [YÖK Ulusal Tez Merkezi](https://tez.yok.gov.tr/UlusalTezMerkezi) · [DergiPark](https://dergipark.org.tr) · [mevzuat HF mirror](https://huggingface.co/datasets/muhammetakkurt/mevzuat-gov-dataset) · [CC language stats](https://commoncrawl.github.io/cc-crawl-statistics/plots/languages)

**Cleaning pipeline:** [FineWeb-2 paper (Turkish canary)](https://arxiv.org/html/2506.20920v1) · [FineWeb paper](https://arxiv.org/pdf/2406.17557) · [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) · [Gopher/MassiveText](https://arxiv.org/pdf/2112.11446) · [CCNet](https://ar5iv.labs.arxiv.org/html/1911.00359) · [datatrove](https://github.com/huggingface/datatrove) · [text-dedup](https://github.com/ChenghaoMou/text-dedup) · [NeMo Curator](https://github.com/chtruong814/NeMo-Curator) · [Dolma](https://arxiv.org/pdf/2402.00159) · [GlotLID](https://arxiv.org/pdf/2310.16248)

**Turkish gotchas:** [Python i/İ casing bug](https://bugs.python.org/issue34723) · [unicode_tr](https://github.com/emre/unicode_tr) · [DeASCIIfication (IPM 2016)](https://www.sciencedirect.com/science/article/abs/pii/S0306457315001053) · [turkish-deasciifier](https://github.com/emres/turkish-deasciifier) · [Turkish tokenization standards](https://arxiv.org/html/2502.07057) · [Tokens with Meaning](https://arxiv.org/html/2508.14292)

**Crawler:** [cc_net](https://github.com/facebookresearch/cc_net) · [ungoliant (OSCAR)](https://github.com/oscar-project/ungoliant) · [UnifiedCrawl](https://arxiv.org/pdf/2411.14343) · [CC Columnar Index](https://commoncrawl.org/columnar-index) · [agmmnn/tr-domains](https://github.com/agmmnn/tr-domains) · [Scrapy vs Crawlee](https://crawlee.dev/blog/scrapy-vs-crawlee) · [WCXB extractor benchmark](https://arxiv.org/html/2605.21097) · [KVKK](https://www.kvkk.gov.tr/Icerik/6649/Personal-Data-Protection-Law) · [EU TDM/LAION ruling](https://www.mofo.com/resources/insights/241004-to-scrape-or-not-to-scrape-first-court-decision)

## Methodology
Four parallel research agents ran ~70 web searches + deep-reads across: (1) existing Turkish corpora inventory, (2) raw sources to reach 1B tokens, (3) cleaning/dedup pipeline + Turkish gotchas, (4) crawler architecture + CC mining + legal. Findings synthesized into the comparison framework and recipe above. Token counts from corpus tables are typically *words*; subword conversions are labeled estimates (×1.3–1.6 for Turkish).
