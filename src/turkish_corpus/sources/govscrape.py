"""Polite scrapers for Turkish government/legal sources (roadmap 2, ``legal`` register).

The HF-mirrored legal source (mevzuat) lives in :mod:`.govlegal`; the sources here have **no
clean dump** and must be fetched from the live state portals, so this module adds real
scrapers built on the polite HTTP client (:class:`turkish_corpus.sources._http.PoliteSession`,
which is throttled, retrying and robots-aware):

- :func:`ingest_resmi_gazete` — the daily official gazette (resmigazete.gov.tr), addressable
  by date. **Implemented** (fixture-tested); the live URL patterns and the daily-index link
  shape must be verified/tuned on the first real run (see below).
- :func:`ingest_tbmm_tutanak` — parliamentary plenary transcripts (tbmm.gov.tr). **Implemented**
  for the HTML path; PDF transcripts reuse :func:`turkish_corpus.sources.academic.extract_pdf_text`.
  Index/URL shapes likewise need live verification.
- :func:`download_court_decisions` — Yargıtay / Danıştay ``karararama`` portals are
  JavaScript-driven and need a browser driver; a documented :exc:`NotImplementedError`
  scaffold with a concrete Playwright/JSON-API plan.
- :func:`download_yoktez` — YÖK Ulusal Tez Merkezi is session + CAPTCHA gated; a documented
  scaffold pointing at the legitimate per-thesis flow + ``ingest_academic --source yoktez``.

LIVE-VERIFICATION WARNING
-------------------------
The author had **no live access to the gov hosts** while writing this, and tests must never
scrape live state sites. So every URL pattern and every HTML-structure assumption lives in a
PURE helper (``build_*_url`` / ``parse_*_index``) unit-tested against fixture HTML strings, and
is clearly marked here as **MUST be verified and tuned on the first real run**. Treat the URL
templates and the link-extraction heuristics as informed guesses, not confirmed contracts.

Heavy deps import lazily: ``trafilatura`` (the ``crawl`` extra) is optional — :func:`_html_to_text`
falls back to a stdlib tag stripper — and ``requests`` (the ``sources`` extra) is only pulled
in by :class:`PoliteSession`. The module therefore imports and is testable under just the
``sources`` extra, with no network.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import date, timedelta
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlsplit

from ._http import DEFAULT_USER_AGENT, PoliteSession
from .base import SourceInfo, make_record, write_records

__all__ = [
    "RESMI_GAZETE",
    "TBMM_TUTANAK",
    "COURT_DECISIONS",
    "YOKTEZ",
    "ingest_resmi_gazete",
    "ingest_tbmm_tutanak",
    "download_court_decisions",
    "download_yoktez",
    "daterange",
    "build_gazette_url",
    "parse_gazette_index",
    "parse_tbmm_index",
]

logger = logging.getLogger(__name__)

# --- Source provenance ----------------------------------------------------------------
# All are official state texts: public, no private copyright holder. The license string is
# carried into every record's metadata so the blend manifest can audit it.
RESMI_GAZETE = SourceInfo(
    name="resmi_gazete",
    license="public (official gazette)",
    register="legal",
    description="Resmî Gazete, the daily Turkish official gazette (resmigazete.gov.tr)",
)
TBMM_TUTANAK = SourceInfo(
    name="tbmm",
    license="public (parliamentary record)",
    register="legal",
    description="TBMM (Grand National Assembly) plenary transcripts (tbmm.gov.tr)",
)
COURT_DECISIONS = SourceInfo(
    name="court_decisions",
    license="public (official text)",
    register="legal",
    description="Turkish high-court decisions (Yargıtay / Danıştay)",
)
YOKTEZ = SourceInfo(
    name="yoktez",
    license="author-permitted / research",
    register="academic",
    description="Turkish master's/PhD theses from YÖK Ulusal Tez Merkezi",
)

# Resmî Gazete archive base. The site serves the historical archive as dated HTML pages under
# /eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm, and a modern date-indexed view at the bare /<YYYYMMDD>
# path. MUST be verified live: the host may redirect, change the path scheme, or serve the
# daily index from a different template than assumed here.
_RG_BASE = "https://www.resmigazete.gov.tr"

# TBMM tutanak portal base. Transcripts are organised per dönem (term) / yasama yılı (legislative
# year) / birleşim (session); the exact index and per-session URLs MUST be verified live.
_TBMM_BASE = "https://www.tbmm.gov.tr"

# Hard cap on a single untrusted page body so a hostile/broken host can't exhaust memory. The
# Content-Length header may be absent (chunked) — that's fine; we only skip when it's oversized.
MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10 MiB


# --- HTML -> text ---------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """A stdlib HTML-to-text fallback used when ``trafilatura`` isn't importable.

    Collects character data while ignoring the contents of ``<script>``/``<style>`` (and other
    non-content) elements, so we never leak JS/CSS into the corpus. ``convert_charrefs`` is on
    (the default) so entities like ``&amp;`` arrive already decoded in ``handle_data``. Block
    boundaries from a small set of structural tags are turned into newlines so the output keeps
    paragraph structure instead of collapsing into one run-on line.
    """

    # Tags whose text content is markup/code, not prose — their data is dropped entirely.
    _SKIP = frozenset({"script", "style", "head", "noscript", "template"})
    # Tags that imply a line/paragraph break in the rendered output.
    _BLOCK = frozenset(
        {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 — stdlib signature
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_tags_stdlib(html: str) -> str:
    """Stdlib HTML→text: strip tags/script/style, normalise whitespace. Never raises.

    Used as the fallback when ``trafilatura`` is unavailable so the module works under just the
    ``sources`` extra. Robust by contract: a malformed document must degrade to "less text",
    never an exception, so the parse is wrapped and a regex tag-strip backs up the parser.
    """
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        text = extractor.get_text()
    except Exception:  # noqa: BLE001 — malformed HTML must never crash a batch ingest
        # Last-ditch: drop script/style blocks then any remaining tags via regex.
        text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _collapse_whitespace(text)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of blank lines/spaces into single separators and trim each line."""
    lines = [line.strip() for line in text.splitlines()]
    # Re-join non-empty lines so paragraph breaks survive but blank runs don't pile up.
    return "\n".join(line for line in lines if line).strip()


def _html_to_text(html: str) -> str:
    """Extract readable text from ``html``: ``trafilatura`` if available, else stdlib fallback.

    ``trafilatura.extract`` (the ``crawl`` extra) gives far better main-content extraction —
    boilerplate/nav stripped — so it is preferred and imported lazily. When it isn't installed
    (module used under just the ``sources`` extra) or returns nothing, we fall back to
    :func:`_strip_tags_stdlib`. Either path is robust: malformed input yields text, never an
    exception, so one bad page can't abort a long ingest.
    """
    if not html:
        return ""
    try:
        import trafilatura  # noqa: PLC0415 — crawl extra; optional

        extracted = trafilatura.extract(html)
        if extracted and extracted.strip():
            return _collapse_whitespace(extracted)
    except ImportError:
        pass  # trafilatura not installed → stdlib fallback below
    except Exception as exc:  # noqa: BLE001 — trafilatura tripping must fall back, not crash
        logger.debug("trafilatura.extract failed; using stdlib fallback: %s", exc)
    return _strip_tags_stdlib(html)


# --- Resmî Gazete (daily official gazette) --------------------------------------------


def daterange(start_date: date, end_date: date) -> Iterator[date]:
    """Yield each :class:`date` from ``start_date`` to ``end_date`` inclusive.

    A generator so a multi-decade range never materialises a giant list. If ``end_date`` is
    before ``start_date`` it yields nothing (an empty range, not an error) — callers treat an
    inverted range as "no work", consistent with the ``limit`` semantics elsewhere.
    """
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_gazette_url(d: date) -> str:
    """Build the Resmî Gazete daily-index URL for date ``d``.

    Uses the historical archive layout ``/eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm``, which has the
    widest date coverage. The modern site also exposes a bare ``/<YYYYMMDD>`` index — switch to
    that if the archive path 404s on the first real run.

    MUST be verified live: the path scheme, zero-padding, and ``.htm`` vs ``.html`` extension
    are informed guesses, not a confirmed contract.
    """
    return f"{_RG_BASE}/eskiler/{d:%Y}/{d:%m}/{d:%Y%m%d}.htm"


def parse_gazette_index(html: str, base_url: str) -> list[str]:
    """Extract item (notice) links from a Resmî Gazete daily-index page. Pure + testable.

    The daily index is a list of anchors pointing at each notice's HTML/PDF. We collect every
    ``href`` and keep those that look like gazette content (``.htm``/``.html``/``.pdf``, or a
    path under the same year archive), resolving relatives against ``base_url`` and preserving
    document order while de-duplicating. The index page itself and obvious non-content anchors
    (mailto/anchors/asset paths) are skipped. Links that resolve off the index's own host are
    dropped (SSRF / off-site link injection guard) so a tampered/compromised index can't redirect
    the crawler to an arbitrary external host.

    MUST be verified live: the real index may wrap links in a known container, use a query-string
    item scheme, or list PDFs only — tune the keep-heuristic against a fetched sample.
    """
    links = _extract_hrefs(html)
    out: list[str] = []
    seen: set[str] = set()
    base_host = urlsplit(base_url).netloc
    for href in links:
        if not href or href.lower().startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute == base_url:
            continue  # self-link
        if urlsplit(absolute).netloc != base_host:
            continue  # off-host link — refuse to follow (SSRF / off-site injection guard)
        lowered = absolute.lower()
        is_doc = lowered.endswith((".htm", ".html", ".pdf"))
        # Same-host archive paths (e.g. /eskiler/YYYY/...) are also content even without an ext.
        same_archive = "/eskiler/" in lowered
        if (is_doc or same_archive) and absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def ingest_resmi_gazete(
    out_dir: str,
    *,
    start_date: date,
    end_date: date,
    session: PoliteSession | None = None,
    limit: int = -1,
) -> int:
    """Scrape Resmî Gazete daily editions in ``[start_date, end_date]`` into raw JSONL.

    For each date: fetch the daily index (:func:`build_gazette_url`), extract notice links
    (:func:`parse_gazette_index`), fetch each notice, convert to text (:func:`_html_to_text`),
    and emit one ``make_record`` per notice with :data:`RESMI_GAZETE` provenance (date + url in
    metadata). Records stream straight into :func:`write_records`; ``limit`` (``<= 0`` = all)
    caps *emitted* records for smoke tests.

    Politeness is delegated to the injected :class:`PoliteSession` (throttle + retry + robots);
    one is created with :data:`DEFAULT_USER_AGENT` if none is passed — but note the CLI guards
    against the placeholder contact before a real run. Network failures on a single date/notice
    are logged and skipped so one bad page can't abort a multi-day crawl.

    LIVE-VERIFICATION: depends on :func:`build_gazette_url` / :func:`parse_gazette_index`, both
    of which encode unverified URL/HTML assumptions — confirm against a real fetch first.
    """
    sess = session or PoliteSession(user_agent=DEFAULT_USER_AGENT)
    return write_records(
        _gazette_records(sess, start_date, end_date, limit),
        out_dir,
    )


def _gazette_records(
    session: PoliteSession,
    start_date: date,
    end_date: date,
    limit: int,
) -> Iterator[dict]:
    """Yield gazette notice records across the date range (see :func:`ingest_resmi_gazete`).

    A generator so notices stream into :func:`write_records` without buffering a whole range in
    memory. ``limit`` counts emitted records; ``0``/negative means no limit.
    """
    emitted = 0
    for d in daterange(start_date, end_date):
        if 0 < limit <= emitted:
            return
        index_url = build_gazette_url(d)
        index_html = _safe_get_text(session, index_url)
        if index_html is None:
            continue  # missing/failed day (e.g. no gazette published) — skip, don't abort
        for i, item_url in enumerate(parse_gazette_index(index_html, index_url)):
            if 0 < limit <= emitted:
                return
            item_html = _safe_get_text(session, item_url)
            if item_html is None:
                continue
            text = _html_to_text(item_html)
            if not text:
                continue
            yield make_record(
                text,
                doc_id=f"rg-{d:%Y%m%d}-{i}",
                source=RESMI_GAZETE,
                date=d.isoformat(),
                url=item_url,
            )
            emitted += 1


# --- TBMM tutanak (parliamentary transcripts) -----------------------------------------


def parse_tbmm_index(html: str) -> list[str]:
    """Extract per-session transcript links from a TBMM tutanak index page. Pure + testable.

    Transcripts are listed as anchors to per-session HTML pages or PDF documents. We collect
    every ``href`` and keep those that look like a tutanak document — an ``.htm``/``.html``/``.pdf``
    link, or a path containing a tutanak marker (``tutanak``/``birlesim``/``donem``) — preserving
    order and de-duplicating. Relative links are returned as-is (the caller knows the index URL
    to resolve them against); absolute links pass through.

    MUST be verified live: the real portal may be a tree of term/year/session pages or expose a
    JSON listing — tune the markers/keep-heuristic against a fetched sample.
    """
    markers = ("tutanak", "birlesim", "birleşim", "donem", "dönem", "yasama")
    out: list[str] = []
    seen: set[str] = set()
    for href in _extract_hrefs(html):
        if not href or href.lower().startswith(("#", "mailto:", "javascript:")):
            continue
        lowered = href.lower()
        is_doc = lowered.endswith((".htm", ".html", ".pdf"))
        has_marker = any(m in lowered for m in markers)
        if (is_doc or has_marker) and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def ingest_tbmm_tutanak(
    out_dir: str,
    *,
    terms,
    session: PoliteSession | None = None,
    limit: int = -1,
) -> int:
    """Scrape TBMM plenary transcripts for the given ``terms`` into raw JSONL; return count.

    ``terms`` is an iterable of dönem (term) identifiers; for each we fetch the term's tutanak
    index (:func:`_tbmm_index_url`), extract per-session links (:func:`parse_tbmm_index`), fetch
    each transcript, convert to text (HTML via :func:`_html_to_text`; PDF via
    :func:`turkish_corpus.sources.academic.extract_pdf_text` after downloading the bytes), and
    emit one ``make_record`` per session with :data:`TBMM_TUTANAK` provenance (term + url in
    metadata). ``limit`` (``<= 0`` = all) caps *emitted* records.

    Politeness via the injected/created :class:`PoliteSession`. Per-session failures are logged
    and skipped. The PDF path imports ``academic.extract_pdf_text`` lazily so the HTML path works
    without the ``academic`` extra.

    LIVE-VERIFICATION: :func:`_tbmm_index_url` / :func:`parse_tbmm_index` encode unverified
    URL/HTML assumptions — confirm against a real fetch first.
    """
    sess = session or PoliteSession(user_agent=DEFAULT_USER_AGENT)
    return write_records(_tbmm_records(sess, terms, limit), out_dir)


def _tbmm_index_url(term) -> str:
    """Build the TBMM tutanak index URL for a legislative ``term`` (dönem).

    MUST be verified live — the real portal's index path/query scheme is unconfirmed; this is a
    placeholder template the first real run must replace.
    """
    query = urlencode({"donem": str(term)})
    return f"{_TBMM_BASE}/Tutanaklar/TutanakSorgu?{query}"


def _tbmm_records(session: PoliteSession, terms, limit: int) -> Iterator[dict]:
    """Yield TBMM transcript records across ``terms`` (see :func:`ingest_tbmm_tutanak`)."""
    emitted = 0
    for term in terms:
        if 0 < limit <= emitted:
            return
        index_url = _tbmm_index_url(term)
        index_html = _safe_get_text(session, index_url)
        if index_html is None:
            continue
        index_host = urlsplit(index_url).netloc
        for i, href in enumerate(parse_tbmm_index(index_html)):
            if 0 < limit <= emitted:
                return
            item_url = urljoin(index_url, href)
            if urlsplit(item_url).netloc != index_host:
                continue  # off-host transcript link — refuse to follow (SSRF guard)
            text = _fetch_tbmm_text(session, item_url)
            if not text:
                continue
            yield make_record(
                text,
                doc_id=f"tbmm-{term}-{i}",
                source=TBMM_TUTANAK,
                term=str(term),
                url=item_url,
            )
            emitted += 1


def _fetch_tbmm_text(session: PoliteSession, url: str) -> str | None:
    """Fetch a transcript and return its text: PDF via academic extractor, else HTML→text.

    PDF transcripts are written to a temp file and handed to
    :func:`turkish_corpus.sources.academic.extract_pdf_text` (lazy import — the ``academic``
    extra is only needed for the PDF path). Any fetch/extract failure returns ``None`` so the
    caller skips the session instead of aborting the term.
    """
    if url.lower().endswith(".pdf"):
        return _fetch_pdf_text(session, url)
    html = _safe_get_text(session, url)
    if html is None:
        return None
    return _html_to_text(html) or None


def _fetch_pdf_text(session: PoliteSession, url: str) -> str | None:
    """Download a PDF and extract its text layer via the academic extractor; ``None`` on failure.

    Writes the bytes to a NamedTemporaryFile because ``extract_pdf_text`` takes a path. The
    extractor is imported lazily so the HTML transcript path doesn't require the ``academic``
    extra. Scanned PDFs (no text layer) return ``None`` from the extractor and are skipped.
    """
    import tempfile  # noqa: PLC0415

    try:
        resp = session.get(url)
    except Exception as exc:  # noqa: BLE001 — network/robots failure: skip this item
        logger.debug("TBMM PDF fetch failed for %s: %s", url, exc)
        return None
    content = getattr(resp, "content", None)
    if not content:
        return None

    from .academic import extract_pdf_text  # noqa: PLC0415 — academic extra; PDF path only

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            return extract_pdf_text(tmp.name)
        except Exception as exc:  # noqa: BLE001 — a bad PDF must not abort the term
            logger.debug("TBMM PDF extract failed for %s: %s", url, exc)
            return None


# --- shared fetch/parse helpers -------------------------------------------------------


def _safe_get_text(session: PoliteSession, url: str) -> str | None:
    """GET ``url`` via the polite session and return ``response.text``, or ``None`` on failure.

    Centralises the "skip a bad page, don't abort the crawl" policy: any network error,
    robots-disallow (:exc:`PermissionError`), non-200 status, or an oversized body (declared
    ``Content-Length`` over :data:`MAX_TEXT_BYTES`) logs at DEBUG and returns ``None`` so the
    caller continues with the next date/item. A missing ``status_code`` defaults to ``0`` (a
    non-2xx value) so a malformed/partial response fails closed rather than being read as 200.
    """
    try:
        resp = session.get(url)
    except Exception as exc:  # noqa: BLE001 — network/robots failure: skip, don't crash batch
        logger.debug("fetch failed for %s: %s", url, exc)
        return None
    status = getattr(resp, "status_code", 0)
    if status != 200:
        logger.debug("non-200 (%s) for %s", status, url)
        return None
    headers = getattr(resp, "headers", None) or {}
    length = headers.get("Content-Length")
    if length is not None:
        try:
            if int(length) > MAX_TEXT_BYTES:
                logger.debug("oversized body (%s bytes) for %s", length, url)
                return None
        except (TypeError, ValueError):
            pass  # unparseable header: fall through and read normally
    return getattr(resp, "text", None)


class _HrefCollector(HTMLParser):
    """Collect every anchor ``href`` in document order. Pure parsing — no network."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 — stdlib signature
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def _extract_hrefs(html: str) -> list[str]:
    """Return all anchor hrefs from ``html`` via the stdlib parser. Never raises.

    A malformed document degrades to whatever was parsed before the error rather than throwing,
    keeping the index-parsing pure helpers robust on real-world (often invalid) gov HTML.
    """
    collector = _HrefCollector()
    try:
        collector.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML: return what we got, don't crash
        pass
    return collector.hrefs


# --- JS/session/CAPTCHA-gated sources: documented scaffolds ----------------------------


def download_court_decisions(
    out_dir: str,
    *,
    session: PoliteSession | None = None,
    limit: int = -1,
) -> int:
    """SCAFFOLD — Yargıtay / Danıştay decisions. JS-driven portal; concrete plan below.

    Portals: Yargıtay (Court of Cassation) https://karararama.yargitay.gov.tr and Danıştay
    (Council of State) https://karararama.danistay.gov.tr. Both are **single-page apps**: the
    search form and result list are rendered client-side, and the decision text is fetched by
    background XHR/fetch calls — so a plain ``PoliteSession.get`` of the page returns an empty
    shell, which is why this is a scaffold rather than a fixture-tested scraper.

    Concrete plan
    -------------
    1. Inspect the network tab on a real search and look for a **JSON API** behind the SPA
       (these portals typically POST a query to an ``/aramalist`` / ``/getDokuman`` style
       endpoint returning JSON with decision ids, then fetch each decision's text/HTML by id).
       If that API exists, call it directly with :class:`PoliteSession` (still throttled +
       robots-aware) — far cheaper and more stable than a browser, and unit-testable with
       fixture JSON exactly like the gazette parser here.
    2. If no usable API is exposed, drive the search UI with **Playwright** (add ``playwright``
       to the ``crawl`` extra): fill the chamber/date-range form, page through results, click
       into each decision, and read the rendered text. Keep the *parsing* of each rendered
       decision in a pure helper so it stays fixture-testable.
    3. Emit one :func:`make_record` per decision with :data:`COURT_DECISIONS`, stamping
       chamber / esas-karar numbers / date into metadata for provenance. Be extremely polite
       (these are state hosts): honour robots, throttle to ~1 req/host, back off on 429/503.

    License: public (official text). Volume: court decisions are the largest legal source by
    far — research puts them at ~3.4B tokens — so shard the output rather than one file.
    """
    raise NotImplementedError(
        "download_court_decisions is a scaffold: the Yargıtay/Danıştay karararama portals are "
        "JS-driven SPAs. Plan: (1) find the JSON search API behind the SPA and call it via "
        "PoliteSession with a fixture-tested JSON parser, OR (2) drive the UI with Playwright "
        "(add to the crawl extra); then emit make_record(..., COURT_DECISIONS) with chamber/"
        "esas-karar/date metadata. ~3.4B tokens — shard the output. See this function's docstring."
    )


def download_yoktez(
    out_pdf_dir: str,
    *,
    session: PoliteSession | None = None,
    limit: int = -1,
) -> int:
    """SCAFFOLD — YÖK Ulusal Tez Merkezi theses. Session + CAPTCHA gated; concrete plan below.

    Portal: https://tez.yok.gov.tr — the National Thesis Center. Bulk automated download is
    **restricted**: the catalogue search and the full-text PDF download are session-gated and
    CAPTCHA-protected, theses can be author-embargoed, and there is no open API or bulk dump.
    Automated mass harvesting is against the portal's terms, so this stays a scaffold by design.

    Concrete plan (legitimate per-thesis flow)
    ------------------------------------------
    1. Search the catalogue for the theses you need and note each thesis number; obtain proper
       access where a thesis is restricted (the per-thesis permission flow), respecting embargoes
       and the author-permitted/research license.
    2. Drive the per-thesis download with **Playwright** (add to the ``crawl`` extra) so a human
       can **solve the CAPTCHA manually** when prompted — one session, minimal concurrency,
       generous delays. Do not attempt to defeat the CAPTCHA programmatically.
    3. Save each permitted full-text PDF into ``out_pdf_dir`` named by thesis number (stable doc
       ids), then feed the directory to the existing extractor::

           uv run --extra academic python scripts/ingest_academic.py \\
               --source yoktez --pdf-dir <out_pdf_dir>

       (i.e. :func:`turkish_corpus.sources.academic.ingest_yoktez`). Many theses are SCANNED —
       plan an OCR pass before re-running extraction (see academic.py's OCR note).

    License: author-permitted / research. This downloader exists to document the boundary; the
    value-add (PDF→records) already lives in :mod:`.academic`.
    """
    raise NotImplementedError(
        "download_yoktez is a scaffold: tez.yok.gov.tr is session + CAPTCHA gated and bulk "
        "download is restricted. Use the legitimate per-thesis flow — drive Playwright (add to "
        "the crawl extra) with MANUAL CAPTCHA solving, save permitted PDFs to out_pdf_dir by "
        "thesis number, then run `ingest_academic --source yoktez` to extract them (many are "
        "scanned and need OCR). See this function's docstring."
    )
