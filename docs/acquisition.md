# Data acquisition (downloaders & scrapers)

How raw text is fetched for the non-HF sources, before the [cleaning pipeline](pipeline.md)
and [blend](blend.md). Every downloader produces either raw `{id,text,metadata}` JSONL (via
the [sources contract](sources.md)) or a directory of PDFs for the academic ingester.

> **No live requests in this codebase's tests.** Parsers are unit-tested against fixtures.
> URL patterns and HTML selectors for government portals are written to documented shapes
> and **must be verified/tuned on first real run** — each is flagged in its docstring.

## Polite HTTP — `sources/_http.py`

`PoliteSession` is the shared client for all downloaders (the Scrapy crawler has its own
politeness). It provides:

- **per-host minimum delay** (sleeps only as long as needed since that host's last request),
- **retry with exponential backoff** on 429/5xx, honouring `Retry-After`,
- an **honest, contactable User-Agent** (the placeholder contact must be replaced — the CLIs
  refuse to run against live hosts until you do, unless `--allow-placeholder-ua`),
- optional **robots.txt** compliance (on for HTML scraping; off for OAI-PMH APIs designed for
  harvesting).

Timing/backoff are pure and unit-tested with injected `sleep`/`clock`; `requests` is lazy.

## DergiPark (academic) — `sources/dergipark.py` ✅

Harvests via **OAI-PMH** (`https://dergipark.org.tr/api/public/oai/` — verify live), the
sanctioned, structured harvest channel, paging through `resumptionToken`. Extracts per-article
PDF URLs from Dublin Core identifiers, downloads PDFs (filename-sanitized, Turkish-preferred)
to a directory, then the [academic ingester](sources.md) extracts text.

```bash
uv run --extra sources python scripts/download_dergipark.py \
    --out-pdf-dir /data/dergipark_pdfs --max-records 5000   # set a real --user-agent
uv run --extra academic python scripts/ingest_academic.py \
    --source dergipark --pdf-dir /data/dergipark_pdfs --out output/raw/dergipark
```

Assumptions to verify live: the OAI base URL + set specs (`?verb=Identify`/`ListSets`), and
the per-article PDF URL shape (we treat `.pdf`/`/download/` identifiers as PDFs). Mostly CC BY
per journal.

## Government / legal — `sources/govscrape.py`

| Source | Status | Notes |
|--------|--------|-------|
| Resmî Gazete | ✅ scraper | Walks the dated archive, parses the daily index, extracts each notice's text. **Verify** the URL pattern (`/eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm` vs modern `/<YYYYMMDD>`) and index link shape. |
| TBMM transcripts | ✅ scraper | Per-term index → HTML (`_html_to_text`) or PDF (reuses academic `extract_pdf_text`). **Verify** the term/session index URL + link markers. |
| Yargıtay/Danıştay courts | 🚧 scaffold | JS-driven `karararama` SPAs — needs a JSON API if one exists, else Playwright (add to the `crawl` extra). ~3.4B tokens (the largest legal source); shard output. |
| YÖKTEZ theses | 🚧 scaffold | Session + CAPTCHA gated; bulk download is restricted. Legitimate per-thesis flow + Playwright with manual CAPTCHA, then `ingest_academic --source yoktez`. |

`_html_to_text(html)` tries `trafilatura` (lazy; via the `crawl` extra) and falls back to a
stdlib `HTMLParser` tag-stripper, so it works under just the `sources` extra.

```bash
uv run --extra sources python scripts/download_govlegal.py \
    --source resmi_gazete --start-date 2024-01-01 --end-date 2024-01-31 --out output/raw/govlegal
uv run --extra sources python scripts/download_govlegal.py --source tbmm --terms 27 28
```

Legislation (mevzuat) is separate — it's a ready HF mirror, no scraping:
`scripts/ingest_govlegal.py`.

## Politeness & legal (KVKK)

- Replace the placeholder `User-Agent` contact with a real, monitored address before any live
  run (the CLIs enforce this).
- Honour robots.txt (default on for HTML); throttle conservatively; back off on 429/503.
- Government texts are **public** (no private copyright); they're the lowest-risk sources.
- The cleaning pipeline still applies **KVKK PII scrubbing** (T.C. Kimlik / phone / IBAN) to
  everything — keep a provenance record (each record's `metadata` carries source/license/url).
- Court and YÖKTEZ portals: respect their access terms; prefer official APIs/bulk channels
  over circumventing session/CAPTCHA gates.

## End-to-end

```
download (this doc) ──▶ raw JSONL / PDFs ──▶ ingest (PDFs→JSONL) ──▶ tc-run-hplt --source jsonl
   ──▶ cleaned JSONL ──▶ build_blend ──▶ final corpus + manifest
```
