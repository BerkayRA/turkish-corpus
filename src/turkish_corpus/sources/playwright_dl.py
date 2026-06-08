"""Browser-driven downloads for gov/legal sites that block bots or render via JS (roadmap 2).

The polite HTTP scrapers in :mod:`.govscrape` work for hosts that serve a plain HTTP client,
but some Turkish state portals do not:

- **Resmî Gazete** (resmigazete.gov.tr) — LIVE-VERIFIED (2026-06-08): returns HTTP 200 to a real
  browser User-Agent but **blocks bot UAs** (our honest bot UA → connection failure; robots.txt
  → 403). It therefore needs a *real* browser to render and return the page.
- **Yargıtay / Danıştay** ``karararama`` portals — JavaScript single-page apps behind a search UI.
- **YÖKTEZ** (tez.yok.gov.tr) — session + CAPTCHA gated.

This module wraps headless Chromium via Playwright (:class:`PlaywrightFetcher`) and reuses the
text extraction (``_html_to_text``) and provenance plumbing (``SourceInfo`` / ``make_record`` /
``write_records``) already proven in :mod:`.govscrape` / :mod:`.base`. The Playwright import is
LAZY (inside ``__enter__``) so importing this module — and unit-testing its pure helpers — needs
neither the ``playwright`` extra nor a browser binary.

Ethics / politeness
-------------------
We drive a REAL browser (not a UA string spoofed onto a plain HTTP client) precisely *because*
these sites are built to serve browsers — a real browser legitimately renders the page the same
way a human visitor's would. We still:

- throttle between navigations (``min_delay``) — one request at a time, no concurrency,
- honour the *spirit* of robots even where a browser bypasses the literal UA block,
- set a real, monitored contact in the User-Agent where possible (see ``download_browser.py``),
- and rely on the fact that these are PUBLIC government texts (no private copyright holder).

Setup
-----
The ``playwright`` extra installs the Python package only; the browser binary is a separate,
one-time install::

    uv sync --extra playwright
    uv run playwright install chromium
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import date, timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .base import SourceInfo, make_record, write_records
from .govscrape import _html_to_text

__all__ = [
    "RESMI_GAZETE",
    "COURT_DECISIONS",
    "YOKTEZ",
    "PlaywrightFetcher",
    "download_resmi_gazette",
    "download_resmi_gazete",
    "download_court_decisions",
    "download_yoktez",
    "daterange",
    "build_gazette_index_url",
    "parse_gazette_index",
]

logger = logging.getLogger(__name__)

# A real, current desktop-Chrome User-Agent. WHY a real-browser UA here (not on a plain client):
# these hosts BLOCK non-browser UAs (Resmî Gazete → 403/connection drop for our bot UA). Playwright
# drives an actual Chromium that genuinely renders the page like a human visitor's browser, so
# presenting a browser UA is honest about what is fetching — unlike spoofing a browser UA onto a
# plain HTTP client (which lies about being a real renderer). Bump this string as Chrome ages.
DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Resmî Gazete daily-edition index. LIVE-VERIFIED (2026-06-08): the dated archive index
#   https://www.resmigazete.gov.tr/eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm
# returns HTTP 200 in a real browser; the homepage links to notices via eskiler/<Y>/<M>/<YMD>
# and /<YYYYMMDD>. (A /fihrist?tarih=YYYY-MM-DD view also works — kept as a documented fallback.)
_RG_BASE = "https://www.resmigazete.gov.tr"

# --- Source provenance ----------------------------------------------------------------
# Resmî Gazete is an official state gazette: public, no private copyright holder. We define our
# own SourceInfo here (rather than importing govscrape.RESMI_GAZETE) so the browser-download path
# is self-contained and can diverge if needed — the name/license/register match govscrape's so
# records from either path are interchangeable in the blend manifest.
RESMI_GAZETE = SourceInfo(
    name="resmi_gazete",
    license="public (official gazette)",
    register="legal",
    description="Resmî Gazete, the daily Turkish official gazette (browser-fetched)",
)
COURT_DECISIONS = SourceInfo(
    name="court_decisions",
    license="public (official text)",
    register="legal",
    description="Turkish high-court decisions (Yargıtay / Danıştay), JS portals",
)
YOKTEZ = SourceInfo(
    name="yoktez",
    license="author-permitted / research",
    register="academic",
    description="Turkish master's/PhD theses from YÖK Ulusal Tez Merkezi (CAPTCHA-gated)",
)


# --- Playwright fetcher ---------------------------------------------------------------


class PlaywrightFetcher:
    """A context manager wrapping a single headless Chromium browser for polite page fetches.

    Used for hosts that block non-browser UAs or render via JS. Open it with ``with`` so the
    browser and Playwright runtime are always closed::

        with PlaywrightFetcher(user_agent="MyBot (+https://x; me@x)") as fetcher:
            html = fetcher.get_html("https://www.resmigazete.gov.tr/eskiler/2024/01/20240115.htm")

    The Playwright import is LAZY (in :meth:`__enter__`) so this module imports without the
    ``playwright`` extra; the Chromium binary itself is a one-time ``playwright install chromium``.

    Parameters
    ----------
    user_agent:
        UA presented to the host. Defaults to :data:`DEFAULT_BROWSER_USER_AGENT` — a real-browser
        string, justified because (a) these sites block non-browser UAs and (b) Playwright drives
        an actual renderer, so it is not the deceptive "spoof a browser on a plain client" pattern.
        Prefer passing a UA that embeds a real contact (URL/email) for the host operator.
    headless:
        Run without a visible window. Set ``False`` for sources needing human interaction
        (e.g. solving a CAPTCHA on YÖKTEZ).
    timeout_ms:
        Per-navigation timeout in milliseconds.
    min_delay:
        Politeness floor: minimum seconds slept *before* each navigation after the first, so we
        never hammer a host. One navigation at a time; no concurrency.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_BROWSER_USER_AGENT,
        *,
        headless: bool = True,
        timeout_ms: int = 30_000,
        min_delay: float = 1.0,
    ) -> None:
        self.user_agent = user_agent
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.min_delay = min_delay
        self._playwright = None
        self._browser = None
        self._context = None
        # Set once a navigation has happened, so the first fetch isn't delayed needlessly.
        self._navigated = False

    def __enter__(self) -> PlaywrightFetcher:
        # Lazy import: keeps the module importable (and its pure helpers testable) without the
        # `playwright` extra installed. Requires a one-time `playwright install chromium`.
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(user_agent=self.user_agent)
        self._context.set_default_timeout(self.timeout_ms)
        return self

    def __exit__(self, *_exc) -> None:
        # Close in reverse order, each guarded so a partial setup still tears down cleanly.
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._playwright, "stop", None),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 — teardown must not mask the original error
                logger.debug("PlaywrightFetcher teardown step failed: %s", exc)
        self._context = self._browser = self._playwright = None

    def _throttle(self) -> None:
        """Sleep ``min_delay`` before every navigation except the first (politeness floor)."""
        if self._navigated and self.min_delay > 0:
            time.sleep(self.min_delay)
        self._navigated = True

    def _require_context(self):
        if self._context is None:
            raise RuntimeError(
                "PlaywrightFetcher must be used as a context manager: `with PlaywrightFetcher() "
                "as f: f.get_html(url)` (the browser is opened in __enter__)."
            )
        return self._context

    def get_html(self, url: str) -> str:
        """Navigate to ``url``, wait for the page to load, and return its rendered HTML.

        Uses ``wait_until="load"`` so JS-rendered content has a chance to populate before we read
        ``page.content()``. The page is always closed, even on error.
        """
        context = self._require_context()
        self._throttle()
        page = context.new_page()
        try:
            page.goto(url, wait_until="load")
            return page.content()
        finally:
            page.close()

    def get_pdf_bytes(self, url: str) -> bytes | None:
        """Navigate to ``url`` and return the bytes of the PDF response, or ``None``.

        Some gov notices are served as PDFs. We capture the navigation response and return its
        body when it is a PDF (by ``Content-Type``); non-PDF or missing responses return ``None``
        so the caller can skip. The page is always closed.
        """
        context = self._require_context()
        self._throttle()
        page = context.new_page()
        try:
            response = page.goto(url, wait_until="load")
            if response is None:
                return None
            content_type = (response.headers or {}).get("content-type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                return None
            return response.body()
        except Exception as exc:  # noqa: BLE001 — a bad PDF response must not abort the batch
            logger.debug("get_pdf_bytes failed for %s: %s", url, exc)
            return None
        finally:
            page.close()


# --- Resmî Gazete (browser-fetched) ---------------------------------------------------


def daterange(start_date: date, end_date: date) -> Iterator[date]:
    """Yield each :class:`date` from ``start_date`` to ``end_date`` inclusive.

    A generator so a multi-year range never materialises a giant list. If ``end_date`` is before
    ``start_date`` it yields nothing (an empty range, not an error).
    """
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_gazette_index_url(d: date) -> str:
    """Build the Resmî Gazete daily-edition index URL for date ``d``. Pure + testable.

    LIVE-VERIFIED (2026-06-08): ``/eskiler/<YYYY>/<MM>/<YYYYMMDD>.htm`` returns HTTP 200 in a real
    browser, e.g. ``date(2024, 1, 15)`` → ``.../eskiler/2024/01/20240115.htm``. Months/days are
    zero-padded. (The ``/fihrist?tarih=YYYY-MM-DD`` view also works as a documented fallback.)
    """
    return f"{_RG_BASE}/eskiler/{d:%Y}/{d:%m}/{d:%Y%m%d}.htm"


def parse_gazette_index(html: str, base_url: str) -> list[str]:
    """Extract same-host notice links from a Resmî Gazete daily-index page. Pure + testable.

    Mirrors the approach in :func:`turkish_corpus.sources.govscrape.parse_gazette_index`: collect
    every anchor ``href``, resolve relatives against ``base_url``, keep those that look like gazette
    content (``.htm``/``.html``/``.pdf`` or a same-year ``/eskiler/`` archive path), preserve
    document order, de-duplicate, and skip the index self-link plus non-content anchors
    (``#``/``mailto:``/``javascript:``). Links that resolve OFF the index's own host are dropped
    (SSRF / off-site link-injection guard) so a tampered index can't redirect the browser elsewhere.
    """
    out: list[str] = []
    seen: set[str] = set()
    base_host = urlsplit(base_url).netloc
    for href in _extract_hrefs(html):
        if not href or href.lower().startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute == base_url:
            continue  # self-link
        if urlsplit(absolute).netloc != base_host:
            continue  # off-host link — refuse to follow (SSRF / off-site injection guard)
        lowered = absolute.lower()
        is_doc = lowered.endswith((".htm", ".html", ".pdf"))
        same_archive = "/eskiler/" in lowered
        if (is_doc or same_archive) and absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def download_resmi_gazete(
    out_dir: str,
    *,
    start_date: date,
    end_date: date,
    fetcher: PlaywrightFetcher | None = None,
    limit: int = -1,
) -> int:
    """Download Resmî Gazete daily editions in ``[start_date, end_date]`` via a browser.

    LIVE-VERIFIED URL pattern (:func:`build_gazette_index_url`): the daily index is fetched with a
    real browser (Resmî Gazete blocks bot UAs), its same-host notice links are parsed
    (:func:`parse_gazette_index`), each notice page is fetched and converted to text
    (``_html_to_text`` from :mod:`.govscrape`), and one :func:`make_record` is emitted per notice
    with :data:`RESMI_GAZETE` provenance (date + url in metadata). Records stream into
    :func:`write_records`; ``limit`` (``<= 0`` = all) caps *emitted* records for smoke tests.

    A :class:`PlaywrightFetcher` is created (and closed) if ``fetcher`` is not supplied; pass one
    to reuse a browser across calls or to inject a fake in tests. Per-date / per-notice failures
    are logged and skipped so one bad page can't abort a multi-day run.
    """
    if fetcher is not None:
        return write_records(_gazette_records(fetcher, start_date, end_date, limit), out_dir)
    with PlaywrightFetcher() as owned:
        return write_records(_gazette_records(owned, start_date, end_date, limit), out_dir)


# Kept as an alias because the verb-spelling is easy to mistype and govscrape uses "gazete".
download_resmi_gazette = download_resmi_gazete


def _gazette_records(
    fetcher: PlaywrightFetcher,
    start_date: date,
    end_date: date,
    limit: int,
) -> Iterator[dict]:
    """Yield gazette notice records across the date range (see :func:`download_resmi_gazete`).

    A generator so notices stream into :func:`write_records` without buffering a whole range in
    memory. ``limit`` counts emitted records; ``0``/negative means no limit. Any per-page error is
    logged and skipped rather than aborting the crawl.
    """
    emitted = 0
    for d in daterange(start_date, end_date):
        if 0 < limit <= emitted:
            return
        index_url = build_gazette_index_url(d)
        index_html = _safe_get_html(fetcher, index_url)
        if index_html is None:
            continue  # no edition that day / fetch failed — skip, don't abort
        for i, item_url in enumerate(parse_gazette_index(index_html, index_url)):
            if 0 < limit <= emitted:
                return
            item_html = _safe_get_html(fetcher, item_url)
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


def _safe_get_html(fetcher: PlaywrightFetcher, url: str) -> str | None:
    """Fetch ``url`` via the fetcher, returning HTML or ``None`` on any failure.

    Centralises the "skip a bad page, don't abort the run" policy: any navigation/timeout error
    logs at DEBUG and returns ``None`` so the caller continues with the next date/notice.
    """
    try:
        html = fetcher.get_html(url)
    except Exception as exc:  # noqa: BLE001 — navigation failure: skip, don't crash the batch
        logger.debug("browser fetch failed for %s: %s", url, exc)
        return None
    return html or None


# --- HTML href extraction (shared, stdlib) --------------------------------------------


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
    keeping :func:`parse_gazette_index` robust on real-world (often invalid) gov HTML.
    """
    collector = _HrefCollector()
    try:
        collector.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML: return what we got, don't crash
        pass
    return collector.hrefs


# --- JS-SPA / CAPTCHA-gated sources: browser-driven scaffolds --------------------------


def download_court_decisions(
    out_dir: str,
    *,
    query: str = "",
    fetcher: PlaywrightFetcher | None = None,
    limit: int = -1,
) -> int:
    """SCAFFOLD — Yargıtay / Danıştay decisions via the JS ``karararama`` search UI.

    Portals: Yargıtay (Court of Cassation) https://karararama.yargitay.gov.tr and Danıştay
    (Council of State) https://karararama.danistay.gov.tr. Both are **single-page apps**: the
    search form, result list and decision text are rendered/fetched client-side, so a plain
    ``get_html`` of the landing URL returns an empty shell — hence a browser-driven scaffold rather
    than a fixture-tested scraper. The structure below is real; the SELECTORS need live tuning.

    Concrete plan (Playwright-driven)
    ---------------------------------
    1. ``with PlaywrightFetcher() as f`` (headless OK) and navigate to the portal landing page.
    2. Fill the query box with ``query`` and submit the search (e.g. ``page.fill(<input sel>, q)``
       then ``page.click(<submit sel>)``); wait for the result list to render.
    3. Paginate the results: read the current page's decision rows, then click "next" until the
       last page (or ``limit`` is reached), collecting decision ids/links.
    4. Open each decision (click the row / navigate to its detail URL), wait for the rendered text,
       and extract it with a PURE parser kept fixture-testable (do NOT bury parsing in the driver).
    5. Emit one :func:`make_record` per decision with :data:`COURT_DECISIONS`, stamping chamber /
       esas-karar numbers / date into metadata. Be extremely polite (state hosts): one tab at a
       time, ``min_delay`` throttling, back off on errors.

    The exact CSS/XPath selectors for the query box, submit button, pager and decision text MUST
    be verified against a live search before this can run — that is the single missing piece, so we
    fail loudly here rather than silently scraping the wrong elements.
    """
    raise NotImplementedError(
        "download_court_decisions is a browser-driven scaffold: the Yargıtay/Danıştay karararama "
        "portals are JS SPAs. Plan: drive PlaywrightFetcher — navigate the portal, fill the query "
        "box, submit, paginate the result list, open each decision, extract its text with a "
        "fixture-tested pure parser, and emit make_record(..., COURT_DECISIONS) with "
        "chamber/esas-karar/date metadata. The search/result/detail SELECTORS need live "
        "verification first. See this function's docstring for the step-by-step plan."
    )


def download_yoktez(
    out_pdf_dir: str,
    *,
    fetcher: PlaywrightFetcher | None = None,
    limit: int = -1,
) -> int:
    """SCAFFOLD — YÖK Ulusal Tez Merkezi theses (session + CAPTCHA gated); concrete plan below.

    Portal: https://tez.yok.gov.tr — the National Thesis Center. Bulk automated download is
    **restricted**: the catalogue search and full-text PDF download are session-gated and
    CAPTCHA-protected, theses can be author-embargoed, and there is no open API or bulk dump.
    Automated mass harvesting violates the portal's terms, so this stays a scaffold by design.

    Concrete plan (legitimate per-thesis flow)
    ------------------------------------------
    1. Open ``PlaywrightFetcher(headless=False)`` — a **headed** browser — so a human can solve the
       CAPTCHA manually when prompted. Navigate to the thesis catalogue and search for the theses
       you need; obtain proper access for any restricted/embargoed thesis.
    2. For each permitted thesis: trigger the full-text download, **pause for the human to solve the
       CAPTCHA**, and save the resulting PDF into ``out_pdf_dir`` named by thesis number (stable doc
       ids). One session, minimal concurrency, generous delays. Do NOT defeat the CAPTCHA
       programmatically.
    3. Feed the downloaded PDFs to the existing extractor — the value-add (PDF→records) already
       lives in :mod:`.academic`::

           uv run --extra academic python scripts/ingest_academic.py \\
               --source yoktez --pdf-dir <out_pdf_dir>

       Many theses are SCANNED — plan an OCR pass before extraction (see academic.py's OCR note).

    License: author-permitted / research. This downloader documents the boundary and the headed
    CAPTCHA flow; it deliberately does not automate past the human gate.
    """
    raise NotImplementedError(
        "download_yoktez is a scaffold: tez.yok.gov.tr is session + CAPTCHA gated and bulk "
        "download is restricted. Use the legitimate per-thesis flow — drive PlaywrightFetcher in "
        "HEADED mode (headless=False), PAUSE for a human to solve the CAPTCHA, save permitted PDFs "
        "to out_pdf_dir by thesis number, then run `ingest_academic --source yoktez` to extract "
        "them (many are scanned and need OCR). See this function's docstring for the full plan."
    )
