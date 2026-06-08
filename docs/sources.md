# Register-diverse sources (roadmap step 2)

A web-only corpus (HPLT) is fluent but narrow. Step 2 adds higher-quality registers and
mixes them to a target token budget. Every source follows the same flow — the cleaning and
dedup are **reused** from steps 1–2, never reimplemented per source:

```
ingest (source-specific)        clean (existing pipeline)         blend (mixer)
─────────────────────────       ─────────────────────────        ─────────────
raw {id,text,metadata} JSONL ──▶ tc-run-hplt --source jsonl   ──▶ build_blend(...)
(sources/<name>/)                 --data-path <raw>                → weighted, budgeted,
                                  → cleaned JSONL (final/)           manifest'd corpus
```

The shared contract is `turkish_corpus.sources.base` (`SourceInfo`, `make_record`,
`write_records`): an ingester just yields `{id, text, metadata:{source,license,register,…}}`
records. Provenance is stamped so the blend [manifest](blend.md) can audit tokens-per-source
and license. Licensing posture for this build: **research/permissive OK** (so OpenSubtitles /
CulturaX are eligible too, though not ingested here yet).

## Implemented ingesters

### Turkish Wikipedia — `sources/wikipedia.py` (`--extra sources`)
Streams the pre-parsed `wikimedia/wikipedia` dataset (clean prose, no wikitext). CC BY-SA
(attribution + share-alike — keep in mind for redistribution).

```bash
uv run --extra sources python scripts/ingest_wikipedia.py \
    --out output/raw/wikipedia --dump 20231101.tr   # verify available dumps on the HF card
```

### Government / legal — `sources/govlegal.py` (`--extra sources`)
`mevzuat` (Turkish legislation) ships now via the `muhammetakkurt/mevzuat-gov-dataset` HF
mirror (the text column is probed at runtime: `text→content→madde→icerik`). **Best license
here** — official texts are public. Resmî Gazete, court decisions (Yargıtay/Danıştay), and
TBMM transcripts are documented **scaffolds** (`NotImplementedError`) — they need polite
scrapers (reuse the [crawler](crawler.md) politeness patterns) and are a later step.

```bash
uv run --extra sources python scripts/ingest_govlegal.py --source mevzuat --out output/raw/govlegal
```

### Academic — `sources/academic.py` (`--extra academic`)
PDF text extraction for **DergiPark** (open-access journals; mostly CC BY per journal) and
**YÖKTEZ** (theses; ~9B+ tokens, highest-quality long-form formal Turkish). `extract_pdf_text`
reads the born-digital text layer (pypdf → pdfplumber fallback) and returns `None` for
**scanned** PDFs, which need an OCR step (tesseract via `ocrmypdf`/`pytesseract`) —
intentionally NOT a dependency. Downloaders are documented scaffolds (DergiPark exposes
OAI-PMH + per-article PDFs; the YÖK portal is session/CAPTCHA-gated — the hard part).

```bash
# After downloading PDFs into a directory:
uv run --extra academic python scripts/ingest_academic.py \
    --source dergipark --pdf-dir /data/dergipark_pdfs --out output/raw/academic
```

## Cleaning each source

Every raw dir is cleaned by the existing pipeline (Turkish normalization, quality filters,
KVKK PII scrubbing, MinHash dedup):

```bash
uv run --extra pipeline tc-run-hplt --source jsonl \
    --data-path output/raw/wikipedia --output output/clean/wikipedia --tasks 8
```

Then mix the cleaned `*/final` dirs with [`build_blend`](blend.md).

## Source status

| Source | Module | Status | License | Register |
|--------|--------|--------|---------|----------|
| Turkish Wikipedia | `wikipedia` | ✅ ingester | CC BY-SA | encyclopedic |
| Legislation (mevzuat) | `govlegal` | ✅ ingester (HF mirror) | public | legal |
| Resmî Gazete / courts / TBMM | `govlegal` | 🚧 scaffold (needs scraper) | public | legal |
| DergiPark | `academic` | ✅ PDF→text (download scaffold) | CC BY (per-journal) | academic |
| YÖKTEZ theses | `academic` | ✅ PDF→text (download scaffold; OCR for scans) | research | academic |
| HPLT web | `pipeline` | ✅ anchor (step 1) | CC0 | web |
