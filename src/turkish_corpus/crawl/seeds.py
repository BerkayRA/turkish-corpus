"""Turn a Common Crawl host-ranking CSV into Scrapy seed inputs.

Pure string/CSV plumbing — no Scrapy, no DuckDB — so the seed parsing that decides what we
crawl is unit-testable and importable without the ``crawl`` extra. The input is the CSV
written by :func:`turkish_corpus.crawl.cc_index.run_host_query`
(``url_host_name,pages`` with a header); the outputs feed
:class:`turkish_corpus.crawl.spider.TurkishSitemapSpider`:

- ``allowed_domains`` — the set Scrapy uses to keep the crawl on-topic (off-site links are
  dropped). We strip a leading ``www.`` because Scrapy's offsite filter matches on the
  registered domain and ``example.com`` already covers ``www.example.com``.
- ``sitemap_urls`` — ``https://<host>/robots.txt`` per host. ``SitemapSpider`` reads
  ``Sitemap:`` directives out of robots.txt, so pointing it at robots.txt both discovers
  the sitemaps *and* respects the same file we obey for politeness.
"""

from __future__ import annotations

import csv

__all__ = [
    "load_hosts",
    "hosts_to_allowed_domains",
    "hosts_to_robots_urls",
]

# Column the CC seed CSV ranks hosts by; used to apply a stricter min_pages floor at load
# time than the query already enforced (handy when reusing one CSV at different cutoffs).
_PAGES_COLUMN = "pages"
_HOST_COLUMN = "url_host_name"


def load_hosts(csv_path: str, *, min_pages: int = 0) -> list[str]:
    """Load host names from a CC seed CSV, optionally filtered by page count.

    Skips the header row, preserves the CSV's existing rank order (the query already sorted
    by ``pages DESC``), and drops blank host cells. When ``min_pages > 0`` and a ``pages``
    column is present, hosts below the threshold are excluded; a missing/non-numeric
    ``pages`` cell is treated as 0 (excluded under any positive floor) so a malformed row
    can never silently inflate the seed list.

    Parameters
    ----------
    csv_path:
        Path to the ``url_host_name,pages`` CSV produced by ``run_host_query``.
    min_pages:
        Minimum page count to keep a host (0 = keep all rows with a non-empty host).
    """
    hosts: list[str] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            host = (row.get(_HOST_COLUMN) or "").strip()
            if not host:
                continue
            if min_pages > 0:
                pages = _to_int(row.get(_PAGES_COLUMN))
                if pages < min_pages:
                    continue
            hosts.append(host)
    return hosts


def hosts_to_allowed_domains(hosts: list[str]) -> list[str]:
    """Map hosts to Scrapy ``allowed_domains``: strip leading ``www.``, dedupe, keep order.

    Order is preserved (first occurrence wins) so the most-pages hosts stay at the front,
    which matters when a caller slices the top-N for a bounded crawl.
    """
    seen: set[str] = set()
    domains: list[str] = []
    for host in hosts:
        domain = _strip_www(host.strip().lower())
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return domains


def hosts_to_robots_urls(hosts: list[str]) -> list[str]:
    """Map hosts to ``https://<host>/robots.txt`` URLs for ``SitemapSpider.sitemap_urls``.

    Hosts are deduped (order preserved) but the ``www.`` prefix is kept as-is: robots.txt is
    served per exact host, so ``www.example.com`` and ``example.com`` can differ and both
    are worth probing when both appear in the seed list.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for host in hosts:
        h = host.strip().lower()
        if h and h not in seen:
            seen.add(h)
            urls.append(f"https://{h}/robots.txt")
    return urls


def _strip_www(host: str) -> str:
    return host[len("www.") :] if host.startswith("www.") else host


def _to_int(value: str | None) -> int:
    try:
        return int(value) if value is not None and value.strip() else 0
    except ValueError:
        return 0
