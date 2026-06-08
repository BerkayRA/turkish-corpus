"""Smoke tests for the trafilatura extraction pipeline (datatrove JSONL hand-off).

These require the optional ``crawl`` extra (Scrapy + trafilatura) and are skipped
otherwise, so the core suite runs without the heavy install. No network is used — the
pipeline is fed an in-memory HTML string.
"""

import pytest

pytest.importorskip("scrapy", reason="needs `uv sync --extra crawl`")
pytest.importorskip("trafilatura", reason="needs `uv sync --extra crawl`")

pytestmark = pytest.mark.crawl

from scrapy.exceptions import DropItem  # noqa: E402

from turkish_corpus.crawl.items import PageItem  # noqa: E402
from turkish_corpus.crawl.pipelines import MIN_TEXT_LENGTH, TrafilaturaPipeline  # noqa: E402

# A short Turkish article with enough body to clear the length floor after extraction.
_PARAGRAPH = (
    "Türkçe doğal dil işleme alanında yüksek kaliteli bir derlem oluşturmak için "
    "morfolojik yapıyı dikkate almak gerekir. Türkçe sondan eklemeli bir dildir ve "
    "kelimeler çok sayıda ek alarak uzar. Bu nedenle çok dilli belirteçleyiciler "
    "Türkçe metinleri gereğinden fazla parçaya böler ve belirteç sayısını şişirir."
)
ARTICLE_HTML = (
    "<html><head><title>Türkçe Derlem</title></head><body>"
    "<nav>menü ana sayfa iletişim</nav>"
    f"<article><h1>Başlık</h1><p>{_PARAGRAPH}</p><p>{_PARAGRAPH}</p></article>"
    "<footer>telif hakkı 2025</footer>"
    "</body></html>"
)


def _make_item(html: str, url: str = "https://example.com.tr/yazi") -> PageItem:
    return PageItem(
        url=url, html=html, title="Türkçe Derlem", fetched_at="2025-06-08T00:00:00+00:00"
    )


class TestTrafilaturaPipeline:
    def test_returns_datatrove_record_shape(self):
        result = TrafilaturaPipeline().process_item(_make_item(ARTICLE_HTML), spider=None)
        assert set(result) == {"id", "text", "metadata"}
        assert result["id"] == "https://example.com.tr/yazi"
        assert isinstance(result["text"], str) and result["text"].strip()
        assert len(result["text"]) >= MIN_TEXT_LENGTH
        assert "Türkçe" in result["text"]
        meta = result["metadata"]
        assert meta["language"] == "tr"
        assert meta["url"] == "https://example.com.tr/yazi"
        assert meta["fetched_at"] == "2025-06-08T00:00:00+00:00"

    def test_drops_too_short_document(self):
        item = _make_item("<html><body><p>Çok kısa.</p></body></html>")
        with pytest.raises(DropItem):
            TrafilaturaPipeline().process_item(item, spider=None)

    def test_drops_missing_html(self):
        item = PageItem(url="https://example.com.tr/x", html="", title=None, fetched_at="")
        with pytest.raises(DropItem):
            TrafilaturaPipeline().process_item(item, spider=None)
