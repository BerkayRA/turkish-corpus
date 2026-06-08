"""Scrapy item pipeline: extract main text and emit a datatrove-ready record.

Importing this module requires the ``crawl`` extra (Scrapy); trafilatura is imported
lazily inside :meth:`TrafilaturaPipeline.process_item` so the rest of the package (and the
unit tests for the pure seed/index helpers) import without the heavy NLP dependency.

This is the join between the crawler and the *existing* cleaning backend. The pipeline
boils a :class:`~turkish_corpus.crawl.items.PageItem` (URL + raw HTML) down to one dict
shaped exactly like a datatrove ``JsonlReader`` record::

    {"id": <url>, "text": <main text>, "metadata": {...}}

``JsonlReader`` defaults to ``text_key="text"`` and ``id_key="id"``, so the JSONL this
emits is read verbatim by ``tc-run-hplt --source jsonl --data-path <crawl_out>``. The
crawler therefore produces *raw* text; all the Turkish normalization, quality filtering,
KVKK PII scrubbing, and dedup are reused from steps 1–2 rather than duplicated here (the
whole point of the DRY hand-off).
"""

from __future__ import annotations

try:
    from scrapy.exceptions import DropItem
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "turkish_corpus.crawl.pipelines requires Scrapy. Install with `uv sync --extra crawl`."
    ) from exc

__all__ = ["TrafilaturaPipeline", "MIN_TEXT_LENGTH"]

# Documents shorter than this (after extraction) are almost always navigation chrome,
# stubs, or boilerplate trafilatura could not strip — not worth a corpus slot. The
# downstream Gopher/FineWeb quality filters would drop most of them anyway; cutting here
# saves writing them at all.
MIN_TEXT_LENGTH = 200


class TrafilaturaPipeline:
    """Extract main text with trafilatura and reshape items into datatrove records.

    Drops (via :class:`scrapy.exceptions.DropItem`) any page with no HTML, no extractable
    text, or text under :data:`MIN_TEXT_LENGTH` characters, so only substantive documents
    reach the JSONL feed.
    """

    def process_item(self, item, spider) -> dict:
        """Turn one :class:`PageItem` into a ``{"id","text","metadata"}`` dict."""
        import trafilatura  # noqa: PLC0415  (crawl extra; lazy to keep imports light)

        url = item.get("url")
        html = item.get("html")
        if not html:
            raise DropItem(f"no HTML body for {url}")

        # favor_precision: prefer dropping borderline boilerplate over keeping noise — we
        # want clean training text, not maximal recall. target_language='tr' nudges the
        # language-aware heuristics; deduplicate drops repeated boilerplate blocks within
        # the page; comments are excluded (forum/comment text is low-quality and PII-dense).
        try:
            text = trafilatura.extract(
                html,
                output_format="txt",
                favor_precision=True,
                include_comments=False,
                target_language="tr",
                deduplicate=True,
            )
        except Exception as exc:  # noqa: BLE001 - never let one bad page stop the crawl
            raise DropItem(f"trafilatura extraction failed for {url}: {exc}") from exc

        if not text or not text.strip():
            raise DropItem(f"no extractable main text for {url}")
        text = text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            raise DropItem(f"text too short ({len(text)} < {MIN_TEXT_LENGTH}) for {url}")

        # datatrove JsonlReader shape: id_key='id', text_key='text'; everything else rides
        # in metadata for provenance and the downstream language gate.
        return {
            "id": url,
            "text": text,
            "metadata": {
                "url": url,
                "title": item.get("title"),
                "language": "tr",
                "fetched_at": item.get("fetched_at"),
                "source": "crawl",
            },
        }
