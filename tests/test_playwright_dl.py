"""Tests for the browser-based downloader — NO real browser (Playwright is mocked).

The ``playwright`` extra installs the package but NOT a browser binary in this env, so these tests
MUST never launch Chromium. Pure helpers (``build_gazette_index_url``, ``daterange``,
``parse_gazette_index``) are exercised against fixture strings; the end-to-end
``download_resmi_gazete`` is driven by a FAKE fetcher; and :class:`PlaywrightFetcher` is tested
with ``sync_playwright`` monkeypatched to a mock so it returns the mocked ``page.content()``.
"""

import gzip
import json
from datetime import date

import pytest

from turkish_corpus.sources import playwright_dl

pytestmark = pytest.mark.playwright


# --- fixtures -------------------------------------------------------------------------


GAZETTE_INDEX_HTML = """
<html><head><style>.x{color:red}</style></head><body>
  <h1>Resmî Gazete</h1>
  <ul>
    <li><a href="20240115-1.htm">Birinci karar</a></li>
    <li><a href="/eskiler/2024/01/20240115-2.htm">İkinci karar</a></li>
    <li><a href="ekler/dosya.pdf">Ek belge</a></li>
    <li><a href="#top">Sayfa başı</a></li>
    <li><a href="mailto:info@example.org">İletişim</a></li>
    <li><a href="https://evil.example.com/off-host.htm">Off-host</a></li>
    <li><a href="20240115-1.htm">Birinci karar (tekrar)</a></li>
  </ul>
</body></html>
"""

GAZETTE_ITEM_HTML = """
<html><head><title>Karar</title><script>var x=1;</script></head>
<body><div><p>Madde 1 — Bu Kanunun amacı kamu düzenini sağlamaktır.</p>
<p>Madde 2 — Yürürlük hükümleri.</p></div></body></html>
"""


class FakeFetcher:
    """A fake PlaywrightFetcher: ``get_html`` returns canned fixtures by URL substring.

    Routing is by substring so tests don't reproduce the exact URL templates the pure helpers
    build. Unknown URLs return ``""`` so the scraper's skip-path is covered. Records calls so
    tests can assert no surprise navigations happened.
    """

    def __init__(self, routes):
        self._routes = routes
        self.calls = []

    def get_html(self, url):
        self.calls.append(url)
        for needle, html in self._routes.items():
            if needle in url:
                return html
        return ""


# --- pure helpers ---------------------------------------------------------------------


def test_build_gazette_index_url_matches_verified_pattern():
    # LIVE-VERIFIED (2026-06-08): /eskiler/YYYY/MM/YYYYMMDD.htm, months/days zero-padded.
    url = playwright_dl.build_gazette_index_url(date(2024, 1, 15))
    assert url == "https://www.resmigazete.gov.tr/eskiler/2024/01/20240115.htm"


def test_daterange_inclusive():
    days = list(playwright_dl.daterange(date(2024, 1, 1), date(2024, 1, 3)))
    assert days == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_daterange_single_day():
    assert list(playwright_dl.daterange(date(2024, 1, 1), date(2024, 1, 1))) == [date(2024, 1, 1)]


def test_daterange_inverted_is_empty():
    assert list(playwright_dl.daterange(date(2024, 1, 3), date(2024, 1, 1))) == []


def test_parse_gazette_index_keeps_same_host_excludes_off_host():
    base = "https://www.resmigazete.gov.tr/eskiler/2024/01/20240115.htm"
    links = playwright_dl.parse_gazette_index(GAZETTE_INDEX_HTML, base)

    # Same-host notice + archive link + pdf, resolved absolute, order-preserving, de-duplicated.
    assert links == [
        "https://www.resmigazete.gov.tr/eskiler/2024/01/20240115-1.htm",
        "https://www.resmigazete.gov.tr/eskiler/2024/01/20240115-2.htm",
        "https://www.resmigazete.gov.tr/eskiler/2024/01/ekler/dosya.pdf",
    ]
    # Off-host, anchors and mailto are excluded.
    assert all("evil.example.com" not in link for link in links)
    assert all(not link.startswith(("#", "mailto:")) for link in links)


def test_parse_gazette_index_malformed_html_does_not_raise():
    base = "https://www.resmigazete.gov.tr/eskiler/2024/01/20240115.htm"
    # Truncated/garbage HTML must degrade, not crash.
    assert playwright_dl.parse_gazette_index("<a href=", base) == []


# --- download_resmi_gazete with a FAKE fetcher (no browser) ----------------------------


def test_download_resmi_gazete_writes_records(tmp_path):
    fetcher = FakeFetcher(
        {
            # The daily index for 2024-01-15.
            "20240115.htm": GAZETTE_INDEX_HTML,
            # Any individual notice page (matches -1.htm / -2.htm / dosya.pdf via substring).
            "20240115-": GAZETTE_ITEM_HTML,
        }
    )
    written = playwright_dl.download_resmi_gazete(
        str(tmp_path),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 15),
        fetcher=fetcher,
    )

    # Two same-host .htm notices yield text; the .pdf routes to "" (no text) and is skipped.
    assert written == 2

    shard = tmp_path / "00000.jsonl.gz"
    assert shard.is_file()
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]

    assert len(records) == 2
    for i, rec in enumerate(records):
        assert rec["id"] == f"rg-20240115-{i}"
        assert "Madde 1" in rec["text"]
        assert rec["metadata"]["source"] == "resmi_gazete"
        assert rec["metadata"]["register"] == "legal"
        assert rec["metadata"]["license"] == "public (official gazette)"
        assert rec["metadata"]["date"] == "2024-01-15"
        assert rec["metadata"]["url"].startswith("https://www.resmigazete.gov.tr/")


def test_download_resmi_gazete_respects_limit(tmp_path):
    fetcher = FakeFetcher(
        {"20240115.htm": GAZETTE_INDEX_HTML, "20240115-": GAZETTE_ITEM_HTML}
    )
    written = playwright_dl.download_resmi_gazete(
        str(tmp_path),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 15),
        fetcher=fetcher,
        limit=1,
    )
    assert written == 1


def test_download_resmi_gazete_skips_missing_day(tmp_path):
    # No route matches the index → "" → skipped, nothing written, run does not crash.
    fetcher = FakeFetcher({})
    written = playwright_dl.download_resmi_gazete(
        str(tmp_path),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 15),
        fetcher=fetcher,
    )
    assert written == 0


# --- PlaywrightFetcher with sync_playwright MONKEYPATCHED (no real browser) -------------


class _MockPage:
    def __init__(self, content):
        self._content = content
        self.closed = False

    def goto(self, url, **_kwargs):
        return None

    def content(self):
        return self._content

    def close(self):
        self.closed = True


class _MockContext:
    def __init__(self, content):
        self._content = content
        self.default_timeout = None

    def set_default_timeout(self, ms):
        self.default_timeout = ms

    def new_page(self):
        return _MockPage(self._content)

    def close(self):
        pass


class _MockBrowser:
    def __init__(self, content):
        self._content = content
        self.headless = None
        self.user_agent = None

    def new_context(self, *, user_agent):
        self.user_agent = user_agent
        return _MockContext(self._content)

    def close(self):
        pass


class _MockChromium:
    def __init__(self, content):
        self._content = content
        self.last_browser = None

    def launch(self, *, headless):
        browser = _MockBrowser(self._content)
        browser.headless = headless
        self.last_browser = browser
        return browser


class _MockPlaywright:
    def __init__(self, content):
        self.chromium = _MockChromium(content)
        self.stopped = False

    def stop(self):
        self.stopped = True


class _MockSyncPlaywrightCM:
    """Stand-in for sync_playwright(): its .start() returns a mock Playwright."""

    def __init__(self, content):
        self._content = content
        self.instance = None

    def start(self):
        self.instance = _MockPlaywright(self._content)
        return self.instance


def _install_mock_playwright(monkeypatch, content):
    """Monkeypatch ``playwright.sync_api.sync_playwright`` to a mock; never launches Chromium."""
    import sys
    import types

    cm = _MockSyncPlaywrightCM(content)
    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: cm
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    return cm


def test_playwright_fetcher_get_html_returns_mocked_content(monkeypatch):
    cm = _install_mock_playwright(monkeypatch, "<html><body>merhaba</body></html>")

    with playwright_dl.PlaywrightFetcher(min_delay=0) as fetcher:
        html = fetcher.get_html("https://example.gov.tr/x")

    assert html == "<html><body>merhaba</body></html>"
    # Teardown ran on exit.
    assert cm.instance.stopped is True


def test_playwright_fetcher_passes_user_agent_and_headless(monkeypatch):
    cm = _install_mock_playwright(monkeypatch, "<html></html>")

    with playwright_dl.PlaywrightFetcher(
        user_agent="MyBot (+https://x; me@x)", headless=False, min_delay=0
    ) as fetcher:
        fetcher.get_html("https://example.gov.tr/x")

    # The actual browser launched by __enter__ recorded headless=False, and its context recorded
    # the UA — proving the constructor params flowed through to the (mocked) Playwright API with
    # no real browser launched.
    browser = cm.instance.chromium.last_browser
    assert browser.headless is False
    assert browser.user_agent == "MyBot (+https://x; me@x)"


def test_playwright_fetcher_requires_context_manager():
    fetcher = playwright_dl.PlaywrightFetcher()
    with pytest.raises(RuntimeError, match="context manager"):
        fetcher.get_html("https://example.gov.tr/x")


# --- scaffolds raise NotImplementedError with helpful guidance -------------------------


def test_download_court_decisions_raises_with_plan(tmp_path):
    with pytest.raises(NotImplementedError) as exc:
        playwright_dl.download_court_decisions(str(tmp_path))
    message = str(exc.value)
    assert "karararama" in message
    assert "Playwright" in message or "PlaywrightFetcher" in message
    assert "selector" in message.lower()


def test_download_yoktez_raises_with_plan(tmp_path):
    with pytest.raises(NotImplementedError) as exc:
        playwright_dl.download_yoktez(str(tmp_path))
    message = str(exc.value)
    assert "tez.yok.gov.tr" in message
    assert "CAPTCHA" in message
    assert "ingest_academic" in message
