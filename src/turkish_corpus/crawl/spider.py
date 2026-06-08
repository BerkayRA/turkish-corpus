"""``TurkishSitemapSpider`` — a polite, sitemap-driven focused crawler.

Importing this module requires the ``crawl`` extra (Scrapy). The spider is intentionally
thin: it discovers pages, fetches HTML, and yields a raw
:class:`~turkish_corpus.crawl.items.PageItem`. All text extraction and the datatrove
reshaping happen in :class:`~turkish_corpus.crawl.pipelines.TrafilaturaPipeline`, and all
politeness lives in :mod:`turkish_corpus.crawl.settings` — so this file stays focused on
"where do I go and what do I keep".

Why ``SitemapSpider``: it reads each host's ``robots.txt``, follows the ``Sitemap:``
directives it finds, and crawls the listed URLs. That means we (a) discover real content
URLs cheaply instead of blindly link-walking, and (b) read the very file we obey for
politeness. The seed host CSV (from CC index mining, step 3a) is loaded at construction
time via the ``hosts_csv`` spider argument and converted with the pure
:mod:`turkish_corpus.crawl.seeds` helpers into ``allowed_domains`` + ``sitemap_urls``.
"""

from __future__ import annotations

from datetime import UTC, datetime

try:
    from scrapy.spiders import SitemapSpider
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "turkish_corpus.crawl.spider requires Scrapy. Install with `uv sync --extra crawl`."
    ) from exc

from .items import PageItem
from .seeds import hosts_to_allowed_domains, hosts_to_robots_urls, load_hosts

__all__ = ["TurkishSitemapSpider"]


class TurkishSitemapSpider(SitemapSpider):
    """Crawl the seed hosts' sitemaps, yielding raw HTML pages as :class:`PageItem`.

    Run with the seed CSV path::

        scrapy crawl turkish_sitemap -a hosts_csv=tr_hosts.csv -a min_pages=100

    or via ``scripts/run_crawl.py``. ``allowed_domains`` and ``sitemap_urls`` are derived
    from the CSV at construction time, so a missing/empty CSV fails fast with a clear error
    rather than silently crawling nothing.
    """

    name = "turkish_sitemap"

    # Only follow sitemap entries that look like HTML content. SitemapSpider matches these
    # regexes against sitemap-listed URLs; ``parse_page`` handles everything kept.
    sitemap_rules = [("", "parse_page")]

    def __init__(self, hosts_csv: str | None = None, min_pages: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hosts_csv:
            raise ValueError(
                "TurkishSitemapSpider requires -a hosts_csv=<path to CC seed CSV>"
            )

        hosts = load_hosts(hosts_csv, min_pages=int(min_pages))
        if not hosts:
            raise ValueError(
                f"no hosts loaded from {hosts_csv!r} (min_pages={min_pages}); "
                "run scripts/cc_seed_query.py first or lower --min-pages"
            )

        # allowed_domains keeps the crawl on the seed hosts; sitemap_urls points
        # SitemapSpider at each host's robots.txt to discover its sitemaps.
        self.allowed_domains = hosts_to_allowed_domains(hosts)
        self.sitemap_urls = hosts_to_robots_urls(hosts)

    def parse_page(self, response):
        """Yield a :class:`PageItem` for one fetched HTML page.

        Skips non-HTML responses defensively (a sitemap can list PDFs/feeds); the heavy
        main-text extraction is the pipeline's job, not the spider's.
        """
        content_type = response.headers.get("Content-Type", b"").decode("latin-1").lower()
        if "html" not in content_type:
            return

        yield PageItem(
            url=response.url,
            html=response.text,
            title=(response.css("title::text").get() or "").strip() or None,
            fetched_at=datetime.now(UTC).isoformat(),
        )
