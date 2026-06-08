"""Tests for the government/legal scrapers — NO network (the session is a fake).

Every URL/HTML assumption is exercised through the PURE helpers (``daterange``,
``build_gazette_url``, ``parse_gazette_index``, ``parse_tbmm_index``) against fixture strings,
and the end-to-end ingest is driven by a fake PoliteSession whose ``.get`` returns canned
fixture HTML — so these tests never touch a live gov host.
"""

import builtins
import gzip
import json
from datetime import date

import pytest

from turkish_corpus.sources import govscrape

pytestmark = pytest.mark.sources


# --- fixtures -------------------------------------------------------------------------


GAZETTE_INDEX_HTML = """
<html><head><style>.x{color:red}</style></head><body>
  <h1>Resmî Gazete</h1>
  <ul>
    <li><a href="20240102-1.htm">Birinci karar</a></li>
    <li><a href="/eskiler/2024/01/20240102-2.htm">İkinci karar</a></li>
    <li><a href="ekler/dosya.pdf">Ek belge</a></li>
    <li><a href="#top">Sayfa başı</a></li>
    <li><a href="mailto:info@example.org">İletişim</a></li>
    <li><a href="20240102-1.htm">Birinci karar (tekrar)</a></li>
  </ul>
</body></html>
"""

GAZETTE_ITEM_HTML = """
<html><head><title>Karar</title><script>var x=1;</script></head>
<body><div><p>Madde 1 — Bu Kanunun amacı kamu düzenini sağlamaktır.</p>
<p>Madde 2 — Yürürlük hükümleri.</p></div></body></html>
"""

TBMM_INDEX_HTML = """
<html><body>
  <a href="/Tutanaklar/birlesim-1.htm">1. Birleşim</a>
  <a href="/Tutanaklar/birlesim-2.pdf">2. Birleşim (PDF)</a>
  <a href="/anasayfa">Ana sayfa</a>
  <a href="#">boş</a>
</body></html>
"""

TBMM_ITEM_HTML = """
<html><body><div><p>BAŞKAN — Birleşimi açıyorum.</p>
<p>Sayın milletvekilleri, gündeme geçiyoruz.</p></div></body></html>
"""


class FakeResponse:
    """Minimal stand-in for a requests.Response (text/content/status_code/headers)."""

    def __init__(self, *, text="", content=b"", status_code=200, headers=None):
        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class FakeSession:
    """A fake PoliteSession: ``.get`` returns canned fixtures by URL, recording calls.

    Routing is by substring so tests don't have to reproduce the exact URL templates the pure
    helpers build. Unknown URLs return a 404 FakeResponse so the scraper's skip-path is covered.
    """

    def __init__(self, routes):
        self._routes = routes
        self.calls = []

    def get(self, url, **_kwargs):
        self.calls.append(url)
        for needle, response in self._routes.items():
            if needle in url:
                return response
        return FakeResponse(status_code=404)


# --- _html_to_text (stdlib fallback) --------------------------------------------------


class TestHtmlToText:
    def test_stdlib_fallback_strips_tags_script_style(self, monkeypatch):
        # Force the trafilatura import to fail so the stdlib fallback is exercised.
        real_import = builtins.__import__

        def no_trafilatura(name, *args, **kwargs):
            if name == "trafilatura":
                raise ImportError("forced: trafilatura unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_trafilatura)

        html = (
            "<html><head><style>.a{color:red}</style>"
            "<script>alert('x')</script></head>"
            "<body><p>Merhaba</p><p>dünya</p></body></html>"
        )
        text = govscrape._html_to_text(html)

        assert "Merhaba" in text
        assert "dünya" in text
        # Script/style content must never leak into the corpus.
        assert "alert" not in text
        assert "color:red" not in text
        # Paragraph boundary preserved as a newline, not collapsed into "Merhabadünya".
        assert "Merhaba\ndünya" in text

    def test_strip_tags_stdlib_direct_handles_malformed_html(self):
        # The stdlib stripper must never raise on broken markup.
        text = govscrape._strip_tags_stdlib("<p>açık <b>kalın metin <i>iç</p")
        assert "açık" in text
        assert "kalın metin" in text

    def test_empty_html_returns_empty(self):
        assert govscrape._html_to_text("") == ""


# --- pure URL/date/index helpers ------------------------------------------------------


class TestDateAndUrlHelpers:
    def test_daterange_inclusive(self):
        days = list(govscrape.daterange(date(2024, 1, 1), date(2024, 1, 3)))
        assert days == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

    def test_daterange_single_day(self):
        assert list(govscrape.daterange(date(2024, 5, 5), date(2024, 5, 5))) == [date(2024, 5, 5)]

    def test_daterange_inverted_yields_nothing(self):
        assert list(govscrape.daterange(date(2024, 1, 3), date(2024, 1, 1))) == []

    def test_build_gazette_url_zero_pads(self):
        url = govscrape.build_gazette_url(date(2024, 1, 2))
        assert url == "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102.htm"


class TestParseGazetteIndex:
    def test_extracts_and_resolves_links(self):
        base = "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102.htm"
        links = govscrape.parse_gazette_index(GAZETTE_INDEX_HTML, base)

        # Relative .htm and absolute archive .htm and the .pdf are kept; resolved to absolute.
        assert "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102-1.htm" in links
        assert "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102-2.htm" in links
        assert "https://www.resmigazete.gov.tr/eskiler/2024/01/ekler/dosya.pdf" in links
        # Anchor (#top) and mailto are dropped.
        assert all("#" not in link and "mailto:" not in link for link in links)
        # De-duplicated: the repeated -1.htm appears once.
        assert links.count("https://www.resmigazete.gov.tr/eskiler/2024/01/20240102-1.htm") == 1


    def test_excludes_off_host_links(self):
        # A tampered/compromised index pointing off-host must not be followed (SSRF guard).
        base = "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102.htm"
        html = (
            '<html><body>'
            '<a href="20240102-1.htm">same-host notice</a>'
            '<a href="http://attacker.com/evil.htm">off-host</a>'
            '<a href="https://evil.example.org/x.pdf">off-host pdf</a>'
            '</body></html>'
        )
        links = govscrape.parse_gazette_index(html, base)

        assert "https://www.resmigazete.gov.tr/eskiler/2024/01/20240102-1.htm" in links
        # Off-host absolute links are dropped even though they end in .htm/.pdf.
        assert all("attacker.com" not in link for link in links)
        assert all("evil.example.org" not in link for link in links)


class TestParseTbmmIndex:
    def test_keeps_transcript_links_only(self):
        links = govscrape.parse_tbmm_index(TBMM_INDEX_HTML)
        assert "/Tutanaklar/birlesim-1.htm" in links
        assert "/Tutanaklar/birlesim-2.pdf" in links
        # The generic /anasayfa link has no doc extension or tutanak marker → dropped.
        assert "/anasayfa" not in links
        assert "#" not in links

    def test_off_host_tbmm_link_not_fetched(self, tmp_path):
        # parse_tbmm_index keeps an absolute off-host .pdf, but the ingest must refuse to fetch it.
        index_html = (
            '<html><body>'
            '<a href="/Tutanaklar/birlesim-1.htm">1. Birleşim</a>'
            '<a href="http://attacker.com/tutanak-evil.htm">off-host tutanak</a>'
            '</body></html>'
        )
        session = FakeSession(
            {
                "donem=27": FakeResponse(text=index_html),
                "birlesim-1.htm": FakeResponse(text=TBMM_ITEM_HTML),
                "attacker.com": FakeResponse(text=TBMM_ITEM_HTML),  # would match if ever fetched
            }
        )
        out_dir = tmp_path / "tbmm"
        written = govscrape.ingest_tbmm_tutanak(
            str(out_dir), terms=["27"], session=session, limit=-1
        )

        # Only the same-host transcript is ingested; the off-host link is skipped before fetch.
        assert written == 1
        assert all("attacker.com" not in url for url in session.calls)


# --- ingest_resmi_gazete end-to-end with a fake session --------------------------------


class TestIngestResmiGazete:
    def test_writes_gzip_jsonl_records(self, tmp_path):
        session = FakeSession(
            {
                # daily index for 20240102
                "20240102.htm": FakeResponse(text=GAZETTE_INDEX_HTML),
                # any notice item
                "20240102-": FakeResponse(text=GAZETTE_ITEM_HTML),
                "dosya.pdf": FakeResponse(text=GAZETTE_ITEM_HTML),
            }
        )
        out_dir = tmp_path / "rg"
        written = govscrape.ingest_resmi_gazete(
            str(out_dir),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            session=session,
            limit=-1,
        )
        # Three distinct notice links (2x .htm + 1x .pdf) → 3 records.
        assert written == 3

        path = out_dir / "00000.jsonl.gz"
        assert path.is_file()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 3
        assert all(r["metadata"]["register"] == "legal" for r in rows)
        assert all(r["metadata"]["source"] == "resmi_gazete" for r in rows)
        assert all(r["metadata"]["date"] == "2024-01-02" for r in rows)
        assert all(r["id"].startswith("rg-20240102-") for r in rows)
        assert all("Madde 1" in r["text"] for r in rows)
        # Turkish chars stored literally, not \\u-escaped.
        assert "düzen".encode() in gzip.decompress(path.read_bytes())

    def test_respects_limit(self, tmp_path):
        session = FakeSession(
            {
                "20240102.htm": FakeResponse(text=GAZETTE_INDEX_HTML),
                "20240102-": FakeResponse(text=GAZETTE_ITEM_HTML),
                "dosya.pdf": FakeResponse(text=GAZETTE_ITEM_HTML),
            }
        )
        out_dir = tmp_path / "rg"
        written = govscrape.ingest_resmi_gazete(
            str(out_dir),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            session=session,
            limit=1,
        )
        assert written == 1

    def test_skips_missing_day(self, tmp_path):
        # An empty route set → every index fetch is a 404 → no records, no crash.
        session = FakeSession({})
        out_dir = tmp_path / "rg"
        written = govscrape.ingest_resmi_gazete(
            str(out_dir),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            session=session,
            limit=-1,
        )
        assert written == 0


# --- ingest_tbmm_tutanak with a fake session ------------------------------------------


class TestIngestTbmmTutanak:
    def test_html_path_writes_records(self, tmp_path):
        session = FakeSession(
            {
                "donem=27": FakeResponse(text=TBMM_INDEX_HTML),
                "birlesim-1.htm": FakeResponse(text=TBMM_ITEM_HTML),
                # the .pdf item has no PDF content route → _fetch_pdf_text returns None, skipped
            }
        )
        out_dir = tmp_path / "tbmm"
        written = govscrape.ingest_tbmm_tutanak(
            str(out_dir),
            terms=["27"],
            session=session,
            limit=-1,
        )
        # Only the HTML transcript yields text; the PDF (no content) is skipped.
        assert written == 1

        path = out_dir / "00000.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert rows[0]["metadata"]["register"] == "legal"
        assert rows[0]["metadata"]["source"] == "tbmm"
        assert rows[0]["metadata"]["term"] == "27"
        assert rows[0]["id"] == "tbmm-27-0"
        assert "BAŞKAN" in rows[0]["text"]


# --- _safe_get_text caps / fail-closed ------------------------------------------------


class TestSafeGetText:
    def test_skips_oversized_body(self):
        # A declared Content-Length over MAX_TEXT_BYTES must be skipped, not read into memory.
        oversized = FakeResponse(
            text="should not be read",
            status_code=200,
            headers={"Content-Length": str(govscrape.MAX_TEXT_BYTES + 1)},
        )
        session = FakeSession({"big": oversized})
        assert govscrape._safe_get_text(session, "https://x.gov.tr/big") is None

    def test_reads_when_within_cap(self):
        ok = FakeResponse(
            text="ok",
            status_code=200,
            headers={"Content-Length": "2"},
        )
        session = FakeSession({"ok": ok})
        assert govscrape._safe_get_text(session, "https://x.gov.tr/ok") == "ok"

    def test_reads_when_content_length_absent(self):
        # No Content-Length header (chunked) is acceptable — still read.
        session = FakeSession({"chunked": FakeResponse(text="body", status_code=200)})
        assert govscrape._safe_get_text(session, "https://x.gov.tr/chunked") == "body"

    def test_missing_status_code_fails_closed(self):
        # A response without status_code defaults to a non-2xx value → treated as failed.
        class _NoStatus:
            text = "body"

        session = FakeSession({"weird": _NoStatus()})
        assert govscrape._safe_get_text(session, "https://x.gov.tr/weird") is None


class TestTbmmIndexUrl:
    def test_url_encodes_term(self):
        # A term needing escaping must be percent-encoded, not interpolated raw.
        url = govscrape._tbmm_index_url("27 & 28")
        assert "donem=27+%26+28" in url
        assert " " not in url


# --- scaffolds ------------------------------------------------------------------------


class TestScaffolds:
    @pytest.mark.parametrize(
        "func",
        [govscrape.download_court_decisions, govscrape.download_yoktez],
    )
    def test_scaffold_raises_with_helpful_message(self, func, tmp_path):
        with pytest.raises(NotImplementedError) as exc:
            func(str(tmp_path))
        message = str(exc.value).lower()
        assert "scaffold" in message
        # The message orients the next implementer on the concrete approach.
        assert "playwright" in message

    def test_court_message_mentions_token_volume(self, tmp_path):
        with pytest.raises(NotImplementedError) as exc:
            govscrape.download_court_decisions(str(tmp_path))
        assert "3.4b" in str(exc.value).lower()

    def test_yoktez_message_points_to_academic_ingest(self, tmp_path):
        with pytest.raises(NotImplementedError) as exc:
            govscrape.download_yoktez(str(tmp_path))
        assert "ingest_academic" in str(exc.value)
