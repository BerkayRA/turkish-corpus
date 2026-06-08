# The crawler (roadmap step 3)

The crawler is an **optional, targeted top-up** for the corpus, not the primary source.
The research verdict is **mine Common Crawl before crawling fresh**: it is cheaper, faster,
and lower legal risk. A fresh crawl only earns its keep for high-value `.tr` content that
Common Crawl (CC) under-covers.

Everything here is gated behind the `crawl` extra:

```bash
uv sync --extra crawl   # duckdb, trafilatura, scrapy, warcio, requests
```

The crawler's only job is to produce **raw** text as datatrove-ready JSONL. All Turkish
normalization, quality filtering, KVKK PII scrubbing, and dedup are **reused** from the
existing cleaning pipeline (steps 1–2) — the crawler does not duplicate any of it.

## Two paths

### Path A — mine Common Crawl (preferred, `3a`)

CC publishes a columnar (Parquet) index of every page it fetched, including per-page
language detection. We query it with DuckDB to rank Turkish hosts by page count — no live
requests, no crawling. That ranked host list is itself useful (an empirical, traffic-
weighted `.tr` seed list) and you can optionally pull individual pages' HTML straight out
of CC's WARC files by byte range (see *Spot-checking* below) instead of re-crawling them.

### Path B — fresh focused crawl (`3b`)

When CC's coverage of a target host is thin, crawl it directly with a polite Scrapy
`SitemapSpider` seeded from the Path A host list. It extracts main text with trafilatura
and writes the same JSONL shape.

## 3a — Common Crawl seed query

```bash
uv run --extra crawl python scripts/cc_seed_query.py \
    --crawl-id CC-MAIN-2025-51 \
    --min-pages 100 --limit 5000 \
    --out tr_hosts.csv
```

This runs (via `turkish_corpus.crawl.cc_index`):

```sql
SELECT url_host_name, COUNT(*) AS pages
FROM read_parquet('s3://commoncrawl/cc-index/table/cc-main/warc/crawl=<ID>/subset=warc/*.parquet')
WHERE fetch_status = 200
  AND content_mime_detected = 'text/html'
  AND content_languages LIKE 'tur%'   -- ISO-639-3; Turkish is 'tur', NOT 'tr'
GROUP BY url_host_name
HAVING pages >= 100
ORDER BY pages DESC
LIMIT 5000
```

Notes:

- **`content_languages` is comma-separated ISO-639-3 in confidence order.** `LIKE 'tur%'`
  keeps primary-Turkish pages. Pass `--all-turkish` for `LIKE '%tur%'` to also include
  pages where Turkish is a *secondary* language (more multilingual noise).
- **Anonymous access**: DuckDB `INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-1';`
  reads the public bucket with no credentials.
- **The crawl id is monthly and version-sensitive.** `CC-MAIN-2025-51` is a default that
  will go stale; **verify the latest snapshot at <https://data.commoncrawl.org>** and pass
  it with `--crawl-id`.
- **Prototype cheaply** with `--parts 'part-00000-*.parquet'` to scan a few index files
  instead of the whole partition before committing to the full (large, slow) scan.

Output is `tr_hosts.csv` (`url_host_name,pages`, ranked).

### Spot-checking a CC record

`turkish_corpus.crawl.range_fetch.fetch_warc_record(warc_filename, offset, length)` pulls a
single page's stored HTML out of CC's WARC files via an HTTP `Range` request (expects
`206`), parsed with `warcio`. Use it to confirm an index row really points at Turkish HTML
before crawling — no full-file download.

## 3b — the focused crawler

```bash
uv run --extra crawl python scripts/run_crawl.py \
    --hosts tr_hosts.csv --out output/crawl --min-pages 100
```

Equivalent to `scrapy crawl turkish_sitemap -a hosts_csv=tr_hosts.csv -a min_pages=100`
with this project's settings loaded. The spider
(`turkish_corpus.crawl.spider.TurkishSitemapSpider`) reads each host's `robots.txt`,
follows the `Sitemap:` directives it finds, fetches HTML pages, and the
`TrafilaturaPipeline` extracts main text and writes JSONL.

### Politeness & compliance (read before any live run)

Settings live in `turkish_corpus/crawl/settings.py`:

- `ROBOTSTXT_OBEY = True` — never fetch a disallowed path.
- AutoThrottle (start 5s, max 60s, target concurrency 1.0), `CONCURRENT_REQUESTS_PER_DOMAIN
  = 1`, `DOWNLOAD_DELAY = 1.0` — at most one in-flight request per host, with backoff.
- Retry on `429`/`503` (and other transient codes) — honour rate-limit/overload signals.
- `DEPTH_LIMIT = 5`, `HTTPCACHE_ENABLED = True` — bounded crawl; dev re-runs hit the cache.
- **`USER_AGENT` carries a contact** — currently the placeholder
  `TurkishCorpusBot/1.0 (+https://example.org/bot; contact@example.org)`. **Set a real,
  monitored URL/email before crawling.**

**KVKK (Turkish data protection):**

- Run all crawler output through the existing pipeline's PII scrubbing (T.C. Kimlik, phone,
  IBAN, email, IP) — it is reused automatically by the hand-off below.
- Comments are excluded at extraction (`include_comments=False`); avoid PII-dense
  forums/community sites when choosing seeds.
- Provenance: every record keeps `metadata.url`, `metadata.fetched_at`, and
  `metadata.source = "crawl"` for auditability.

## The hand-off: crawler JSONL → existing cleaning pipeline

Each emitted line is exactly a datatrove `JsonlReader` record:

```json
{"id": "<url>", "text": "<extracted main text>", "metadata": {"url": "...", "title": "...", "language": "tr", "fetched_at": "...", "source": "crawl"}}
```

`JsonlReader` defaults (`text_key="text"`, `id_key="id"`) read it verbatim, so the crawl
output is cleaned by the **same** pipeline as steps 1–2:

```bash
uv run --extra pipeline tc-run-hplt \
    --source jsonl --data-path output/crawl \
    --tokenizer /models/tr-morph/tokenizer.json \
    --output /data/corpus/crawl_tur --tasks 8
```

(Consider `--fix-mojibake` for fresh crawl text, since unlike HPLT it is not pre-ftfy'd.)
This DRY reuse is the whole point: the crawler adds a source; the curation backend stays
single-sourced.
```
