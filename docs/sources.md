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

### Government / legal — `sources/govlegal.py` + `sources/govscrape.py` (`--extra sources`)
`mevzuat` (Turkish legislation) ships via the `muhammetakkurt/mevzuat-gov-dataset` HF mirror
(text column probed at runtime: `text→content→madde→icerik`) — no scraping. **Best license
here** — official texts are public.

```bash
uv run --extra sources python scripts/ingest_govlegal.py --out output/raw/govlegal
```

The scraped legal sources live in `govscrape.py` (`scripts/download_govlegal.py`) on the
[polite HTTP client](acquisition.md): **Resmî Gazete** and **TBMM transcripts** are
implemented (verify URL patterns live); **courts** (Yargıtay/Danıştay) and **YÖKTEZ** are
documented scaffolds (JS/CAPTCHA-gated — need Playwright). See [`docs/acquisition.md`](acquisition.md).

### Academic — `sources/academic.py` (`--extra academic`)
PDF text extraction for **DergiPark** (open-access journals; mostly CC BY per journal) and
**YÖKTEZ** (theses; ~9B+ tokens, highest-quality long-form formal Turkish). `extract_pdf_text`
reads the born-digital text layer (pypdf → pdfplumber fallback) and returns `None` for
**scanned** PDFs, which need an OCR step (tesseract via `ocrmypdf`/`pytesseract`) —
intentionally NOT a dependency.

DergiPark now has a real **OAI-PMH downloader** (`sources/dergipark.py`,
`scripts/download_dergipark.py`) — see [`docs/acquisition.md`](acquisition.md). YÖKTEZ
download stays a scaffold (session/CAPTCHA-gated).

```bash
# Download PDFs (DergiPark OAI-PMH), then extract text:
uv run --extra sources python scripts/download_dergipark.py --out-pdf-dir /data/dergipark_pdfs
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
| Resmî Gazete | `playwright_dl` | ⚠️ Playwright (blocks bot UAs; URL verified) | public | legal |
| TBMM transcripts | `govscrape` | ✅ scraper (verify URLs; 302 redirect) | public | legal |
| Yargıtay/Danıştay courts | `playwright_dl` | 🚧 Playwright skeleton | public | legal |
| YÖKTEZ download | `playwright_dl` | 🚧 Playwright skeleton (manual CAPTCHA) | research | academic |
| DergiPark | `dergipark`+`academic` | ✅ OAI-PMH (live-verified) + PDF→text | CC BY (per-journal) | academic |
| YÖKTEZ theses | `academic` | ✅ PDF→text (download scaffold; OCR for scans) | research | academic |
| HPLT web | `pipeline` | ✅ anchor (step 1) | CC0 | web |
