"""Tests for the DergiPark OAI-PMH harvester + PDF downloader — no network, fixture XML only.

OAI XML is parsed from fixture strings and the network is faked (a tiny fake session whose
``.get`` returns objects with ``.text`` / ``.content``), so nothing here touches a live host.
"""

from __future__ import annotations

import pytest

pytest.importorskip("requests")  # sources extra; skip cleanly if it's not installed

from turkish_corpus.sources import dergipark  # noqa: E402

pytestmark = pytest.mark.sources


# A two-record ListRecords page. dc:identifier holds the realistic DergiPark article *landing*
# URL plus an izlik.org permalink (verified live 2026-06-08: neither is a direct PDF URL). The
# second record has no article URL at all. A resumptionToken keeps paging going. Namespaces match
# the real OAI/oai_dc envelope.
_PAGE_WITH_TOKEN = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header><identifier>oai:dergipark.org.tr:article/111</identifier></header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Turkce Makale Bir</dc:title>
          <dc:language>tr</dc:language>
          <dc:identifier>https://dergipark.org.tr/en/pub/J/article/10</dc:identifier>
          <dc:identifier>https://izlik.org/JA73JR34SZ</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <record>
      <header><identifier>oai:dergipark.org.tr:article/222</identifier></header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Makale Iki</dc:title>
          <dc:language>tr</dc:language>
          <dc:identifier>https://izlik.org/ZZ99ZZ99ZZ</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken>TOKEN-PAGE-2</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""

# A final page: one record, empty resumptionToken element (OAI convention for "end").
_PAGE_NO_TOKEN = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header><identifier>oai:dergipark.org.tr:article/333</identifier></header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Makale Uc</dc:title>
          <dc:language>en</dc:language>
          <dc:identifier>https://dergipark.org.tr/en/pub/J/article/33</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken></resumptionToken>
  </ListRecords>
</OAI-PMH>
"""

# A realistic article-page HTML fragment carrying BOTH the full-text PDF anchor
# (/tr/download/article-file/9) and a citation-download anchor
# (/tr/download/article-cite-file/10/type/2). The resolver must pick the former, skip the latter.
# Verified live 2026-06-08: the file id (9) differs from the article id (10).
_ARTICLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="tr">
  <body>
    <a class="cite" href="/tr/download/article-cite-file/10/type/2">Atif (BibTeX)</a>
    <a class="article-file" href="/tr/download/article-file/9">Tam Metin (PDF)</a>
  </body>
</html>
"""


class _FakeResponse:
    """A minimal stand-in for ``requests.Response`` exposing ``.text``/``.content``/headers.

    ``iter_content`` streams ``content`` in fixed-size chunks so the streaming PDF download path
    can be exercised without a real ``requests.Response``.
    """

    def __init__(
        self, *, text: str = "", content: bytes = b"", headers: dict[str, str] | None = None
    ) -> None:
        self.text = text
        self.content = content
        self.headers = headers or {}

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class _FakeSession:
    """A fake :class:`PoliteSession` whose ``.get`` replays queued responses by call order.

    Records the URLs it was asked for so tests can assert paging semantics without a network.
    Accepts and ignores ``**kwargs`` (e.g. ``stream=True``) like the real ``PoliteSession.get``.
    """

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.urls.append(url)
        return self._responses.pop(0)


# A billion-laughs / entity-expansion payload: a tiny document that, if entities were expanded,
# blows up to gigabytes. A hardened parser must refuse to expand it.
_BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
  <record><header><identifier>&lol3;</identifier></header></record>
</ListRecords></OAI-PMH>
"""


class TestXmlEntityExpansion:
    def test_billion_laughs_does_not_expand(self):
        # With defusedxml installed the parse must raise EntitiesForbidden; the stdlib fallback
        # (CPython >= 3.12.7) simply leaves the entity unexpanded. Either way, no GB-scale blowup.
        defused = pytest.importorskip("defusedxml.ElementTree")
        with pytest.raises(defused.EntitiesForbidden):
            dergipark.parse_listrecords(_BILLION_LAUGHS)


class TestParseListRecords:
    def test_parses_records_and_token(self):
        records, token = dergipark.parse_listrecords(_PAGE_WITH_TOKEN)

        assert token == "TOKEN-PAGE-2"
        assert len(records) == 2

        first, second = records
        assert first["identifier"] == "oai:dergipark.org.tr:article/111"
        assert first["title"] == "Turkce Makale Bir"
        assert first["language"] == "tr"
        # The DergiPark article landing URL is picked — not the izlik.org permalink, and not a
        # (non-existent) direct PDF URL.
        assert first["article_url"] == "https://dergipark.org.tr/en/pub/J/article/10"
        # Raw Dublin Core is retained for downstream provenance.
        assert first["dc"]["title"] == ["Turkce Makale Bir"]

        # The second record has only an izlik.org identifier → no article_url.
        assert second["title"] == "Makale Iki"
        assert second["article_url"] is None

    def test_empty_token_is_none(self):
        records, token = dergipark.parse_listrecords(_PAGE_NO_TOKEN)
        assert token is None
        assert len(records) == 1


class TestHarvestRecords:
    def test_pages_across_resumption_token(self):
        session = _FakeSession(
            [
                _FakeResponse(text=_PAGE_WITH_TOKEN),  # page 1: 2 records + token
                _FakeResponse(text=_PAGE_NO_TOKEN),  # page 2: 1 record, no token → stop
            ]
        )

        out = list(dergipark.harvest_records(session, max_records=-1))

        # All records across both pages are yielded.
        assert [r["identifier"] for r in out] == [
            "oai:dergipark.org.tr:article/111",
            "oai:dergipark.org.tr:article/222",
            "oai:dergipark.org.tr:article/333",
        ]
        # First request carries metadataPrefix; the second carries only the resumptionToken.
        assert "verb=ListRecords&metadataPrefix=oai_dc" in session.urls[0]
        assert "resumptionToken=TOKEN-PAGE-2" in session.urls[1]
        assert "metadataPrefix" not in session.urls[1]

    def test_max_records_stops_early(self):
        session = _FakeSession([_FakeResponse(text=_PAGE_WITH_TOKEN)])

        out = list(dergipark.harvest_records(session, max_records=1))

        # Stops after the budget without requesting page 2.
        assert len(out) == 1
        assert len(session.urls) == 1

    def test_set_spec_is_passed(self):
        session = _FakeSession([_FakeResponse(text=_PAGE_NO_TOKEN)])
        list(dergipark.harvest_records(session, set_spec="journal-x"))
        assert "&set=journal-x" in session.urls[0]


class _DownloadSession:
    """A fake session for the two-step resolve-then-download flow; records requested URLs.

    An ``/article/`` URL returns the article-page HTML (so :func:`resolve_pdf_url` can extract the
    ``/download/article-file/<id>`` link); any other URL (the resolved PDF) returns streamed PDF
    bytes. Accepts ``**kwargs`` (``stream=True``) like the real session, and the PDF response
    supports ``iter_content`` so the streaming + size-cap path is exercised. ``content`` is
    configurable so oversized-PDF behaviour can be tested.
    """

    def __init__(
        self, content: bytes = b"%PDF-1.4 fake", *, article_html: str = _ARTICLE_PAGE_HTML
    ) -> None:
        self.urls: list[str] = []
        self._content = content
        self._article_html = article_html

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.urls.append(url)
        if "/article/" in url:
            return _FakeResponse(text=self._article_html)
        return _FakeResponse(content=self._content)


class TestExtractArticleFileHref:
    def test_picks_article_file_over_cite_file(self):
        href = dergipark.extract_article_file_href(
            _ARTICLE_PAGE_HTML, "https://dergipark.org.tr/tr/pub/J/article/10"
        )
        # The full-text article-file link is resolved to an absolute URL; the cite-file is skipped.
        assert href == "https://dergipark.org.tr/tr/download/article-file/9"

    def test_returns_none_when_no_article_file(self):
        html = '<a href="/tr/download/article-cite-file/10/type/2">cite only</a>'
        assert (
            dergipark.extract_article_file_href(html, "https://dergipark.org.tr/x") is None
        )


class TestResolvePdfUrl:
    def test_resolves_to_absolute_pdf_url(self):
        session = _DownloadSession()
        url = dergipark.resolve_pdf_url(
            session, "https://dergipark.org.tr/en/pub/J/article/10"
        )
        assert url == "https://dergipark.org.tr/tr/download/article-file/9"

    def test_returns_none_when_page_has_no_pdf(self):
        session = _DownloadSession(article_html="<html><body>no link here</body></html>")
        url = dergipark.resolve_pdf_url(
            session, "https://dergipark.org.tr/en/pub/J/article/10"
        )
        assert url is None


class TestDownloadPdfs:
    def test_resolves_and_writes_records_with_article_url(self, tmp_path):
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/111",
                "language": "tr",
                "article_url": "https://dergipark.org.tr/en/pub/J/article/10",
            },
            {  # no article_url → skipped (never fetched)
                "identifier": "oai:dergipark.org.tr:article/222",
                "language": "tr",
                "article_url": None,
            },
        ]
        session = _DownloadSession()

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 1
        written = sorted(p.name for p in tmp_path.glob("*.pdf"))
        # Filename is the sanitised identifier (no path separators / colons survive).
        assert written == ["oai_dergipark.org.tr_article_111.pdf"]
        assert (tmp_path / written[0]).read_bytes() == b"%PDF-1.4 fake"
        # The article page was fetched, then the resolved PDF URL — two GETs for the one record.
        assert session.urls == [
            "https://dergipark.org.tr/en/pub/J/article/10",
            "https://dergipark.org.tr/tr/download/article-file/9",
        ]

    def test_skips_records_with_unresolvable_page(self, tmp_path):
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/111",
                "language": "tr",
                "article_url": "https://dergipark.org.tr/en/pub/J/article/10",
            }
        ]
        session = _DownloadSession(article_html="<html><body>no pdf</body></html>")

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 0
        assert list(tmp_path.glob("*.pdf")) == []
        # Only the article page was fetched; no PDF download was attempted.
        assert session.urls == ["https://dergipark.org.tr/en/pub/J/article/10"]

    def test_skips_non_turkish_records(self, tmp_path):
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/333",
                "language": "en",  # explicit non-Turkish → skipped (never fetched)
                "article_url": "https://dergipark.org.tr/en/pub/J/article/33",
            },
            {
                "identifier": "oai:dergipark.org.tr:article/444",
                "language": None,  # unknown language → kept
                "article_url": "https://dergipark.org.tr/tr/pub/J/article/44",
            },
        ]
        session = _DownloadSession()

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 1
        assert [p.name for p in tmp_path.glob("*.pdf")] == [
            "oai_dergipark.org.tr_article_444.pdf"
        ]
        # The non-Turkish record was never even fetched.
        assert "https://dergipark.org.tr/en/pub/J/article/33" not in session.urls

    def test_limit_caps_downloads(self, tmp_path):
        records = [
            {
                "identifier": f"oai:dergipark.org.tr:article/{i}",
                "language": "tr",
                "article_url": f"https://dergipark.org.tr/tr/pub/J/article/{i}",
            }
            for i in range(5)
        ]
        session = _DownloadSession()

        count = dergipark.download_pdfs(records, tmp_path, session, limit=2)

        assert count == 2
        assert len(list(tmp_path.glob("*.pdf"))) == 2


class TestSafePdfUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/x.pdf",  # cloud IMDS (link-local)
            "http://10.0.0.1/x.pdf",  # RFC-1918 private (10/8)
            "http://192.168.1.5/x.pdf",  # RFC-1918 private (192.168/16)
            "http://172.16.0.1/x.pdf",  # RFC-1918 private (172.16/12)
            "http://localhost/x.pdf",  # loopback hostname
            "http://127.0.0.1/x.pdf",  # loopback IP
            "https://evil.com/x.pdf",  # off-allowlist host
            "ftp://dergipark.org.tr/x.pdf",  # wrong scheme even on an allowed host
            "file:///etc/passwd",  # no host / wrong scheme
        ],
    )
    def test_rejects_unsafe_urls(self, url):
        assert dergipark._is_safe_pdf_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://dergipark.org.tr/tr/download/article-file/9.pdf",
            "https://www.dergipark.org.tr/download/article-file/10.pdf",
            "http://dergipark.org.tr/download/article-file/11.pdf",
        ],
    )
    def test_accepts_allowlisted_dergipark_urls(self, url):
        assert dergipark._is_safe_pdf_url(url) is True

    def test_download_skips_ssrf_url(self, tmp_path):
        # A (hostile/compromised) article page that resolves the PDF link to the cloud metadata
        # endpoint must be caught by the SSRF guard: the page is fetched, but the PDF is not.
        article_url = "https://dergipark.org.tr/tr/pub/J/article/evil"
        evil_html = (
            '<a href="http://169.254.169.254/latest/download/article-file/1">PDF</a>'
        )
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/evil",
                "language": "tr",
                "article_url": article_url,
            }
        ]
        session = _DownloadSession(article_html=evil_html)

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 0
        # The article page was fetched, but the resolved (unsafe) PDF URL was never requested.
        assert session.urls == [article_url]
        assert list(tmp_path.glob("*.pdf")) == []

    def test_download_skips_off_allowlist_host(self, tmp_path):
        article_url = "https://dergipark.org.tr/tr/pub/J/article/evil2"
        evil_html = '<a href="https://evil.com/tr/download/article-file/1">PDF</a>'
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/evil2",
                "language": "tr",
                "article_url": article_url,
            }
        ]
        session = _DownloadSession(article_html=evil_html)

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 0
        assert session.urls == [article_url]
        assert list(tmp_path.glob("*.pdf")) == []


class TestPdfSizeCap:
    def test_oversized_pdf_is_skipped(self, tmp_path):
        # A PDF whose streamed body exceeds MAX_PDF_BYTES must be aborted and not written.
        big = b"x" * (dergipark.MAX_PDF_BYTES + 1)
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/huge",
                "language": "tr",
                "article_url": "https://dergipark.org.tr/tr/pub/J/article/huge",
            }
        ]
        session = _DownloadSession(content=big)

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 0
        assert list(tmp_path.glob("*.pdf")) == []

    def test_within_cap_pdf_is_written(self, tmp_path):
        records = [
            {
                "identifier": "oai:dergipark.org.tr:article/ok",
                "language": "tr",
                "article_url": "https://dergipark.org.tr/tr/pub/J/article/ok",
            }
        ]
        session = _DownloadSession(content=b"%PDF-1.4 small")

        count = dergipark.download_pdfs(records, tmp_path, session)

        assert count == 1
        written = list(tmp_path.glob("*.pdf"))
        assert len(written) == 1
        assert written[0].read_bytes() == b"%PDF-1.4 small"


class TestFilenameSanitisation:
    def test_rejects_path_traversal(self):
        # A malicious identifier must never escape the output directory.
        name = dergipark._safe_filename("../../etc/passwd")
        assert "/" not in name
        assert not name.startswith("..")
        assert name.endswith(".pdf")

    def test_caps_long_identifier(self):
        # A 300-char identifier must not produce a filename over the 255-byte fs limit.
        name = dergipark._safe_filename("a" * 300)
        assert name.endswith(".pdf")
        assert len(name) <= 255
        # The slug is capped at _MAX_SLUG_LEN, leaving room for the ".pdf" suffix.
        assert len(name) == dergipark._MAX_SLUG_LEN + len(".pdf")

    def test_traversal_record_stays_contained(self, tmp_path):
        records = [
            {
                "identifier": "../../../evil",
                "language": "tr",
                "article_url": "https://dergipark.org.tr/tr/pub/J/article/9",
            }
        ]
        session = _DownloadSession()

        dergipark.download_pdfs(records, tmp_path, session)

        # Exactly one file, written inside tmp_path (no parent escaped).
        written = list(tmp_path.glob("*.pdf"))
        assert len(written) == 1
        assert written[0].parent == tmp_path
