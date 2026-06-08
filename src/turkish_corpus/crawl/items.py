"""The single Scrapy item carried from spider to extraction pipeline.

Importing this module requires the ``crawl`` extra (Scrapy). The spider yields a raw
:class:`PageItem` (URL + fetched HTML + cheap metadata); :class:`~turkish_corpus.crawl.
pipelines.TrafilaturaPipeline` turns that HTML into clean main text and reshapes it into
the datatrove JSONL record. Keeping the raw HTML on the item (not extracting in the spider)
lets the politeness/download concern stay separate from the CPU-bound extraction concern,
and keeps the spider easy to test.
"""

from __future__ import annotations

try:
    import scrapy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "turkish_corpus.crawl.items requires Scrapy. Install with `uv sync --extra crawl`."
    ) from exc

__all__ = ["PageItem"]


class PageItem(scrapy.Item):
    """A fetched HTML page awaiting main-text extraction.

    Fields
    ------
    url:
        Final (post-redirect) URL of the page; becomes the datatrove record ``id``.
    html:
        Raw decoded HTML body, handed to trafilatura for extraction.
    title:
        Best-effort ``<title>`` text, carried into the record metadata.
    fetched_at:
        ISO-8601 UTC timestamp of fetch, for provenance (KVKK auditability).
    """

    url = scrapy.Field()
    html = scrapy.Field()
    title = scrapy.Field()
    fetched_at = scrapy.Field()
