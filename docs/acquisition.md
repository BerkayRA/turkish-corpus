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

## DergiPark (academic) — `sources/dergipark.py` ✅ (live-verified 2026-06-08)

Harvests via **OAI-PMH** (`https://dergipark.org.tr/api/public/oai/`), the sanctioned,
structured harvest channel, paging through `resumptionToken`. **Verified live:** the OAI
`dc:identifier` is the article *landing* URL (e.g. `.../pub/mulkiye/article/10`), **not** a
PDF, so the downloader resolves the PDF in a second step — fetch the article page and extract
its `/<lang>/download/article-file/<id>` link (excluding `article-cite-file` citation links),
then download. PDFs are filename-sanitized, Turkish-preferred (via `dc:language`), SSRF-guarded
(dergipark host allowlist), and size-capped. The [academic ingester](sources.md) then extracts
text. Confirmed end-to-end against a real article (`article/10` → `download/article-file/9`).

```bash
uv run --extra sources python scripts/download_dergipark.py \
    --out-pdf-dir /data/dergipark_pdfs --max-records 5000   # set a real --user-agent
uv run --extra academic python scripts/ingest_academic.py \
    --source dergipark --pdf-dir /data/dergipark_pdfs --out output/raw/dergipark
```

Assumptions to verify live: the OAI base URL + set specs (`?verb=Identify`/`ListSets`), and
the per-article PDF URL shape (we treat `.pdf`/`/download/` identifiers as PDFs). Mostly CC BY
per journal.

## Government / legal — `sources/govscrape.py` (plain HTTP) + `sources/playwright_dl.py` (browser)

| Source | Status | Notes |
|--------|--------|-------|
| Resmî Gazete | ⚠️ **use Playwright** | **Verified live 2026-06-08:** resmigazete.gov.tr returns 200 to a *browser* UA but blocks bot UAs (our honest bot UA → connection failure; robots.txt → 403). The plain-HTTP `govscrape.ingest_resmi_gazete` is therefore blocked from most clients; use `playwright_dl.download_resmi_gazete` (real browser). **Verified URL pattern:** `/eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm` (and `/fihrist?tarih=YYYY-MM-DD`). |
| TBMM transcripts | ✅ scraper (verify) | `govscrape.ingest_tbmm_tutanak`: per-term index → HTML (`_html_to_text`) or PDF (academic `extract_pdf_text`). Live probe: the tutanak landing 302-redirects; the term/session index URL + link markers still need live tuning. |
| Yargıtay/Danıştay courts | 🚧 Playwright skeleton | `playwright_dl.download_court_decisions`: JS `karararama` SPAs — documented Playwright plan (navigate→search→paginate→extract), selectors need live tuning. ~3.4B tokens (largest legal source); shard output. |
| YÖKTEZ theses | 🚧 Playwright skeleton | `playwright_dl.download_yoktez`: session + CAPTCHA gated; headed-mode + manual CAPTCHA, then PDFs → `ingest_academic --source yoktez`. |

`_html_to_text(html)` tries `trafilatura` (lazy; via the `crawl` extra) and falls back to a
stdlib `HTMLParser` tag-stripper, so it works under just the `sources` extra.

```bash
# Browser-based (Resmî Gazete blocks bot UAs); needs a one-time `playwright install chromium`:
uv run --extra playwright python scripts/download_browser.py \
    --source resmi_gazete --start-date 2024-01-01 --end-date 2024-01-31 --out output/raw/govlegal

# TBMM via plain HTTP (verify URL patterns live):
uv run --extra sources python scripts/download_govlegal.py --source tbmm --terms 27 28
```

Legislation (mevzuat) is separate — a ready HF mirror, no scraping: `scripts/ingest_govlegal.py`.

## Browser-based downloads — `sources/playwright_dl.py` (`--extra playwright`)

For sites that block non-browser UAs (Resmî Gazete) or render via JS/are CAPTCHA-gated
(courts, YÖKTEZ), `PlaywrightFetcher` drives a real headless Chromium. This is the honest
path: rather than *spoofing* a browser UA on a plain HTTP client to evade a block, a real
browser legitimately renders the page. Still throttled, still honours the spirit of robots,
and government texts are public. Requires a one-time `playwright install chromium`.

`download_resmi_gazete` is a real implementation (verified URL pattern); courts/YÖKTEZ are
documented Playwright skeletons whose selectors need live tuning.

## Verification status (probed 2026-06-08)

| Endpoint | Result |
|----------|--------|
| DergiPark OAI-PMH | ✅ verified; PDF resolved via article page (`article/10`→`download/article-file/9`) |
| Resmî Gazete | ⚠️ blocks bot UAs (403/blocked); 200 with browser → Playwright path; URL pattern confirmed |
| TBMM tutanak | ⚠️ 302 redirect; index structure needs live tuning |
| Yargıtay/Danıştay, YÖKTEZ | not probed (JS/CAPTCHA) — Playwright skeletons |

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
