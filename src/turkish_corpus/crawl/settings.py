"""Polite Scrapy settings for the Turkish focused crawler (roadmap step 3b).

Module-level constants so Scrapy can ``getdict``/import them and so ``scripts/run_crawl.py``
can hand them to a ``CrawlerProcess``. The defaults are deliberately conservative: this
crawler tops up a corpus, it is not a hurry. Politeness here protects both the target hosts
and the project's standing (KVKK + robots compliance — see ``docs/crawler.md``).

Key choices and *why*:

- ``ROBOTSTXT_OBEY`` — never fetch a path a host disallowed. Non-negotiable.
- AutoThrottle + ``CONCURRENT_REQUESTS_PER_DOMAIN = 1`` + ``DOWNLOAD_DELAY`` — at most one
  in-flight request per host with a real gap, backing off automatically when a host slows.
- Retry on 429/503 — honour rate-limit / overload signals instead of hammering.
- ``HTTPCACHE_ENABLED`` — re-running the crawl during development re-reads cached responses
  instead of re-hitting hosts.
- ``USER_AGENT`` — identifies the bot and a contact. The contact is a **placeholder**;
  set a real, monitored address before any live run (see docs/crawler.md).
"""

from __future__ import annotations

# Identify the bot honestly and give hosts a way to reach a human. PLACEHOLDER contact —
# replace with a real monitored URL/email before crawling for real.
USER_AGENT = "TurkishCorpusBot/1.0 (+https://example.org/bot; contact@example.org)"

# --- Politeness -----------------------------------------------------------------------
ROBOTSTXT_OBEY = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5.0
AUTOTHROTTLE_MAX_DELAY = 60.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 1.0

# --- Retry / backoff ------------------------------------------------------------------
RETRY_ENABLED = True
RETRY_TIMES = 2
# Defaults plus the rate-limit (429) and overload (503) codes worth a polite retry.
RETRY_HTTP_CODES = [429, 500, 502, 503, 504, 522, 524, 408]

# --- Scope / depth --------------------------------------------------------------------
DEPTH_LIMIT = 5
# A single host with a huge sitemap can expand the crawl far past its share. Cap total
# pages per run (0 = unlimited); raise deliberately once you've sized a host's budget.
CLOSESPIDER_PAGECOUNT = 50_000

# --- Caching (dev re-runs hit the cache, not the hosts) -------------------------------
# NOTE: the on-disk HTTP cache stores fetched pages UNENCRYPTED and may contain personal
# data — treat the httpcache/ dir with the same KVKK obligations as the output and exclude
# it from long-term storage. Expire after a day so production re-runs don't replay stale
# (or since-removed) pages indefinitely.
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86_400  # 1 day

# --- Pipeline wiring ------------------------------------------------------------------
ITEM_PIPELINES = {
    "turkish_corpus.crawl.pipelines.TrafilaturaPipeline": 300,
}

# Forward-compatible Scrapy defaults; quiet a couple of deprecation warnings.
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# --- Output ---------------------------------------------------------------------------
# One JSONL line per item, datatrove JsonlReader shape (see pipelines.py). Overridden at
# runtime by scripts/run_crawl.py to a per-run path; this default keeps `scrapy crawl`
# usable out of the box.
FEEDS = {
    "output/crawl/%(name)s_%(time)s.jsonl": {
        "format": "jsonlines",
        "encoding": "utf-8",
        "overwrite": False,
    },
}
