"""Crawler tooling (roadmap step 3): mine Common Crawl, then crawl fresh — politely.

Two capabilities, both feeding the *existing* datatrove cleaning backend (steps 1–2):

- **3a — Common Crawl index seed query** (:mod:`~turkish_corpus.crawl.cc_index`): a DuckDB
  query over CC's columnar Parquet index ranks the highest-yield Turkish hosts into a seed
  CSV, with no live requests.
- **3b — Scrapy focused crawler** (:mod:`~turkish_corpus.crawl.spider`,
  :mod:`~turkish_corpus.crawl.pipelines`, :mod:`~turkish_corpus.crawl.settings`): a polite
  sitemap-driven crawl over those hosts that extracts main text with trafilatura and writes
  datatrove-ready JSONL (``{"id","text","metadata"}``), cleaned by
  ``tc-run-hplt --source jsonl``.

Only the *pure* helpers are re-exported here so ``import turkish_corpus.crawl`` works
without the ``crawl`` extra (Scrapy/DuckDB/trafilatura). The Scrapy/DuckDB-bound modules
(``items``, ``pipelines``, ``settings``, ``spider``, ``range_fetch``, and
``cc_index.run_host_query``) import their heavy dependencies lazily and are imported on
demand. See ``docs/crawler.md``.
"""

from __future__ import annotations

from .cc_index import (
    CC_INDEX_PATH_TEMPLATE,
    DEFAULT_CRAWL_ID,
    build_index_path,
    build_tr_host_query,
)
from .seeds import hosts_to_allowed_domains, hosts_to_robots_urls, load_hosts

__all__ = [
    "CC_INDEX_PATH_TEMPLATE",
    "DEFAULT_CRAWL_ID",
    "build_index_path",
    "build_tr_host_query",
    "load_hosts",
    "hosts_to_allowed_domains",
    "hosts_to_robots_urls",
]
