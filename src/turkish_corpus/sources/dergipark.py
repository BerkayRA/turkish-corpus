"""DergiPark OAI-PMH harvester + PDF downloader (roadmap 2, ``academic`` register).

DergiPark (https://dergipark.org.tr) hosts thousands of Turkish open-access journals and
exposes an **OAI-PMH** endpoint — the standard, polite interface *designed* for harvesting
metadata. Rather than scrape article pages, we harvest Dublin Core records (title, language,
identifiers) over OAI-PMH. OAI-PMH is built for this, so the session disables robots.txt (it is
the sanctioned bulk channel), but we still throttle and back off via :class:`PoliteSession`.

PDF RESOLUTION FLOW (verified live 2026-06-08)
----------------------------------------------
The OAI ``dc:identifier`` values are **not** direct PDF links — they are the article *landing*
URL (``https://dergipark.org.tr/<lang>/pub/<journal>/article/<id>``) and an ``izlik.org``
permalink, neither of which ends in ``.pdf`` or routes through ``/download/``. The real PDF
link only appears on the article page. So resolution is a two-step flow:

1. OAI gives the article landing URL (:func:`_best_article_url`, recorded as ``article_url``).
2. Fetch that article page and extract the first ``/<lang>/download/article-file/<id>`` anchor
   (:func:`resolve_pdf_url` / :func:`extract_article_file_href`), **excluding** the
   ``/<lang>/download/article-cite-file/...`` citation links that also appear on the page. The
   file id differs from the article id (e.g. article ``/10`` → file ``/9``).
3. Download that ``/download/article-file/<id>`` URL (it is on dergipark.org.tr, so it passes
   the :func:`_is_safe_pdf_url` SSRF allowlist) under the streamed :data:`MAX_PDF_BYTES` cap.

This module only **downloads** PDFs. Turning them into datatrove-ready JSONL is the existing,
offline extraction step in :mod:`.academic` (``ingest_dergipark`` / the ``ingest_academic``
CLI), which the pipeline then cleans. Download is split from extraction because fetching is
the network-bound, portal-specific part; extraction is pure-local.

ASSUMPTIONS TO VERIFY LIVE
--------------------------
- The OAI base URL is :data:`OAI_BASE` (``/api/public/oai/``). DergiPark's exact endpoint and
  the available ``set`` specs (per-journal / per-publisher) should be confirmed against a live
  ``?verb=Identify`` / ``?verb=ListSets`` request before a real harvest — they are not pinned
  here and no network is touched in tests or at import.

LICENSE
-------
DergiPark articles are mostly **CC BY per-journal** — verify each journal's terms rather than
assume. The conservative license string lives on :data:`academic.DERGIPARK` and is stamped
into every record's metadata by the extraction step.

Heavy/optional deps (``requests`` via :class:`PoliteSession`) import lazily. XML parsing uses
``defusedxml`` (the ``sources`` extra) to harden against entity-expansion (billion-laughs)
attacks on the untrusted OAI XML, falling back to stdlib ``xml.etree.ElementTree`` — which is
itself safe against entity expansion on CPython >= 3.12.7 (expat 2.6.3+ ships BLAP) — when
``defusedxml`` isn't installed, so the module still imports without the ``sources`` extra.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET  # Element type hints only; parsing goes via _xml_fromstring
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

try:
    from defusedxml.ElementTree import fromstring as _xml_fromstring
except ImportError:  # stdlib is safe on CPython >= 3.12.7 (expat 2.6.3+ has BLAP)
    from xml.etree.ElementTree import fromstring as _xml_fromstring

from ._http import DEFAULT_USER_AGENT, PoliteSession

__all__ = [
    "OAI_BASE",
    "parse_listrecords",
    "harvest_records",
    "extract_article_file_href",
    "resolve_pdf_url",
    "download_pdfs",
    "download_dergipark",
]

logger = logging.getLogger(__name__)

# DergiPark's public OAI-PMH endpoint. VERIFY LIVE (see module docstring): confirm the URL and
# available set specs with ?verb=Identify / ?verb=ListSets before a real harvest.
OAI_BASE = "https://dergipark.org.tr/api/public/oai/"

# OAI-PMH and Dublin Core namespaces. OAI-PMH wraps each record's metadata in oai_dc, whose
# fields live in the Dublin Core elements namespace; we register both to find elements.
_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# Turkish language codes we keep when a record declares its language (ISO 639-1 / 639-2).
# Records without a language are kept (DergiPark is Turkish-majority; downstream language ID
# filters in the pipeline catch any non-Turkish slip-through).
_TURKISH_LANGS = frozenset({"tr", "tur"})

# A dc:identifier is the article landing URL when it is on a DergiPark host and contains an
# /article/ path segment (e.g. https://dergipark.org.tr/tr/pub/<journal>/article/<id>). The
# other identifier is an izlik.org permalink, which we ignore. Verified live 2026-06-08.
_ARTICLE_URL = re.compile(r"https?://(?:www\.)?dergipark\.org\.tr/.*?/article/\d+", re.IGNORECASE)

# The full-text PDF anchor on an article page: /<lang>/download/article-file/<file-id>. We must
# NOT match /<lang>/download/article-cite-file/... (citation downloads), so "article-file" is
# anchored to a path boundary. Verified live 2026-06-08.
_ARTICLE_FILE_HREF = re.compile(
    r"""href\s*=\s*["'](?P<href>/[^/"']+/download/article-file/\d+[^"']*)["']""",
    re.IGNORECASE,
)

# Hard size caps on untrusted responses so a hostile/broken endpoint can't exhaust memory or
# disk: a single OAI page's XML and a single article PDF. Generous enough for real DergiPark
# content (articles are a few MB; OAI pages a few hundred KB) but bounded.
MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10 MiB cap on a ListRecords XML page
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MiB cap on a single article PDF

# Cap the sanitised filename slug so "<slug>.pdf" stays under the 255-byte fs filename limit.
_MAX_SLUG_LEN = 240

# SSRF allowlist: pdf_url values are resolved from untrusted article-page HTML, so we only fetch
# from the known DergiPark hosts. EXTEND LIVE if PDFs are served from a verified CDN subdomain
# (e.g. a cdn.dergipark.org.tr) — add it here only after confirming it against a real response.
_PDF_HOST_ALLOWLIST = frozenset({"dergipark.org.tr", "www.dergipark.org.tr"})

# Host substrings/prefixes that resolve to the local machine or private networks (SSRF / cloud
# metadata exfiltration). We reject these defensively even though the allowlist already blocks
# them, so the intent is explicit and the helper is reusable if the allowlist is ever relaxed.
_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# RFC-1918 private + link-local (169.254.*, which includes the 169.254.169.254 cloud IMDS).
_PRIVATE_HOST_RE = re.compile(
    r"^(?:"
    r"169\.254\."  # link-local / cloud instance-metadata service
    r"|10\."  # 10.0.0.0/8
    r"|192\.168\."  # 192.168.0.0/16
    r"|172\.(?:1[6-9]|2\d|3[01])\."  # 172.16.0.0/12
    r")"
)


def _is_safe_pdf_url(url: str) -> bool:
    """Whether ``url`` is safe to fetch as an article PDF (SSRF guard on untrusted OAI data).

    ``url`` is resolved from an untrusted article page's HTML (:func:`resolve_pdf_url`), so a
    hostile/compromised page could point it at ``http://169.254.169.254/...`` (cloud metadata),
    ``localhost``, or an internal RFC-1918 host. We require an ``http``/``https`` scheme, reject
    loopback / link-local / private hosts, and enforce an allowlist of known DergiPark hosts
    (:data:`_PDF_HOST_ALLOWLIST`). Returns ``False`` (skip + log) for anything that fails — fail
    closed. The expected ``/download/article-file/<id>`` URL is on dergipark.org.tr, so it passes.
    """
    try:
        split = urlsplit(url)
    except ValueError:
        return False
    if split.scheme not in ("http", "https"):
        return False
    host = (split.hostname or "").lower()
    if not host:
        return False
    if host in _BLOCKED_HOSTS or _PRIVATE_HOST_RE.match(host):
        return False
    return host in _PDF_HOST_ALLOWLIST


def _dc_fields(record: ET.Element) -> dict[str, list[str]]:
    """Collect a record's Dublin Core fields into ``{tag: [values...]}`` (stripped, non-empty).

    DC fields repeat (multiple identifiers, creators, languages), so each maps to a *list*.
    Returns ``{}`` for a deleted/header-only record with no metadata block, which the caller
    treats as "nothing to harvest" rather than an error.
    """
    fields: dict[str, list[str]] = {}
    for elem in record.iter():
        # ElementTree tags are "{namespace}local"; we only want Dublin Core elements.
        if not elem.tag.startswith("{http://purl.org/dc/elements/1.1/}"):
            continue
        local = elem.tag.split("}", 1)[1]
        value = (elem.text or "").strip()
        if value:
            fields.setdefault(local, []).append(value)
    return fields


def _best_article_url(identifiers: list[str]) -> str | None:
    """Pick the DergiPark article landing URL from a record's dc:identifier values.

    DergiPark records advertise the article landing URL
    (``https://dergipark.org.tr/<lang>/pub/<journal>/article/<id>``) and an ``izlik.org``
    permalink — neither is a direct PDF link (verified live 2026-06-08). We return the first
    identifier on a dergipark.org.tr host whose path contains ``/article/<digits>`` (ignoring the
    izlik.org permalink), or ``None`` when no such URL is present. The PDF itself is resolved
    from the article page later via :func:`resolve_pdf_url`.
    """
    for ident in identifiers:
        if _ARTICLE_URL.match(ident.strip()):
            return ident.strip()
    return None


def extract_article_file_href(html: str, base_url: str) -> str | None:
    """Extract the full-text PDF href from an article page's HTML, as an absolute URL.

    DergiPark article pages link the PDF as ``/<lang>/download/article-file/<file-id>`` and also
    expose ``/<lang>/download/article-cite-file/<id>/type/<n>`` *citation* links — we want the
    former and must skip the latter (verified live 2026-06-08; the file id differs from the
    article id). Returns the first matching ``article-file`` href joined against ``base_url`` into
    an absolute URL, or ``None`` when the page has no such anchor. Pure/testable: regex over the
    raw HTML, no network and no new deps.
    """
    for match in _ARTICLE_FILE_HREF.finditer(html):
        href = match.group("href")
        # _ARTICLE_FILE_HREF anchors "/download/article-file/" so "article-cite-file" never
        # matches; this guard is belt-and-braces in case the pattern is ever loosened.
        if "article-cite-file" in href.lower():
            continue
        return urljoin(base_url, href)
    return None


def resolve_pdf_url(session: PoliteSession, article_url: str) -> str | None:
    """Resolve an article landing URL to its full-text PDF URL by fetching the article page.

    GETs ``article_url`` via the (polite, throttled) ``session``, then parses the returned HTML
    for the ``/<lang>/download/article-file/<id>`` anchor (see :func:`extract_article_file_href`),
    returning it as an absolute URL on the DergiPark host. Any fetch error, an over-cap page, or a
    page with no article-file link yields ``None`` (the caller logs + skips that record) so one
    unresolvable article never aborts the batch.
    """
    try:
        resp = session.get(article_url)
    except Exception as exc:  # noqa: BLE001 — one bad fetch must not abort the harvest
        logger.warning("failed to fetch article page %s: %s", article_url, exc)
        return None
    html = _capped_text(resp, article_url)
    if html is None:
        return None
    return extract_article_file_href(html, article_url)


def _parse_record(record: ET.Element) -> dict:
    """Build one harvested record dict from an OAI ``<record>`` element.

    Keys: ``identifier`` (OAI header id), ``title``, ``language``, ``article_url`` (the DergiPark
    article landing URL; the PDF is resolved from it later via :func:`resolve_pdf_url`), and
    ``dc`` (the raw Dublin Core ``{tag: [values]}`` map for downstream provenance). First value
    wins for the scalar fields; ``article_url``/``language`` are ``None`` when absent.
    """
    header_id = record.findtext("oai:header/oai:identifier", default="", namespaces=_NS).strip()
    dc = _dc_fields(record)
    titles = dc.get("title", [])
    languages = dc.get("language", [])
    identifiers = dc.get("identifier", [])
    return {
        "identifier": header_id,
        "title": titles[0] if titles else None,
        "language": languages[0] if languages else None,
        "article_url": _best_article_url(identifiers),
        "dc": dc,
    }


def parse_listrecords(xml_text: str) -> tuple[list[dict], str | None]:
    """Parse an OAI ``ListRecords`` response into ``(records, resumption_token)``.

    Pure and stdlib-only so it is unit-testable on fixture XML with no network. Returns the
    parsed record dicts (see :func:`_parse_record`) and the ``<resumptionToken>`` text for
    paging, or ``None`` when the token is absent/empty (last page). An empty resumptionToken
    element (the OAI convention for "you reached the end") is normalised to ``None``.
    """
    root = _xml_fromstring(xml_text)
    records = [_parse_record(rec) for rec in root.findall(".//oai:record", _NS)]
    token_elem = root.find(".//oai:resumptionToken", _NS)
    token = (token_elem.text or "").strip() if token_elem is not None else ""
    return records, (token or None)


def _capped_text(resp, url: str) -> str | None:
    """Return ``resp.text`` unless its declared ``Content-Length`` exceeds :data:`MAX_TEXT_BYTES`.

    Guards against an unbounded read of an untrusted OAI page. The ``Content-Length`` header may
    be absent (chunked responses) — that's acceptable; we only cap when the server declares an
    oversized body. Returns ``None`` (caller skips/stops) when the cap is exceeded.
    """
    headers = getattr(resp, "headers", None) or {}
    length = headers.get("Content-Length")
    if length is not None:
        try:
            if int(length) > MAX_TEXT_BYTES:
                logger.warning("OAI page exceeds %d bytes, skipping: %s", MAX_TEXT_BYTES, url)
                return None
        except (TypeError, ValueError):
            pass  # unparseable header: fall through and read normally
    return getattr(resp, "text", None)


def harvest_records(
    session: PoliteSession,
    *,
    metadata_prefix: str = "oai_dc",
    set_spec: str | None = None,
    max_records: int = -1,
) -> Iterator[dict]:
    """Harvest DergiPark OAI ``ListRecords``, following ``resumptionToken`` paging; yield dicts.

    The first request carries ``verb=ListRecords&metadataPrefix=...`` (plus ``&set=`` when a
    ``set_spec`` is given); every subsequent request carries only
    ``verb=ListRecords&resumptionToken=...`` per the OAI spec (the token already encodes the
    prefix/set). Yields one parsed record dict per ``<record>`` and stops after ``max_records``
    when ``max_records > 0`` (``-1`` = harvest until the token is exhausted).

    A generator so callers stream straight into :func:`download_pdfs` without materialising the
    whole harvest. ``session.get(...).text`` supplies the XML; parsing is delegated to the pure
    :func:`parse_listrecords` so this loop is just paging + budget control.
    """
    params = f"verb=ListRecords&metadataPrefix={quote(metadata_prefix, safe='')}"
    if set_spec:
        params += f"&set={quote(set_spec, safe='')}"
    url = f"{OAI_BASE}?{params}"

    emitted = 0
    while True:
        resp = session.get(url)
        xml_text = _capped_text(resp, url)
        if xml_text is None:
            return  # oversized/missing OAI page — stop paging rather than risk OOM
        records, token = parse_listrecords(xml_text)
        for record in records:
            if 0 < max_records <= emitted:
                return
            yield record
            emitted += 1
        if token is None:
            return
        # Subsequent pages: the token alone identifies the request (no prefix/set repeated).
        url = f"{OAI_BASE}?verb=ListRecords&resumptionToken={quote(token, safe='')}"


def _safe_filename(identifier: str) -> str:
    """Turn an OAI identifier into a traversal-safe ``<slug>.pdf`` filename.

    OAI identifiers look like ``oai:dergipark.org.tr:article/12345`` and can contain ``/``,
    ``:``, and ``..`` — all unsafe for a path component. We keep only ``[A-Za-z0-9._-]`` and
    collapse the rest to ``_``, then strip any leading dots so the name can never start a
    parent-directory traversal (``..``) or a hidden/relative path. Empty results fall back to
    a constant so we always produce a valid, contained filename. The slug is capped at 240 chars
    so the final ``<slug>.pdf`` stays within the 255-byte filename limit on common filesystems.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).lstrip(".")[:_MAX_SLUG_LEN]
    if not slug:
        slug = "record"
    return f"{slug}.pdf"


def _is_turkish(record: dict) -> bool:
    """Whether a record should pass the Turkish-language filter.

    Keeps records whose declared language is Turkish (``tr``/``tur``) and *also* records with
    no declared language at all (DergiPark is Turkish-majority and many records omit the field;
    the pipeline's language-ID filter is the real backstop). Only an explicit non-Turkish
    language is rejected here.
    """
    language = record.get("language")
    if not language:
        return True
    return language.strip().lower() in _TURKISH_LANGS


def download_pdfs(
    records: Iterable[dict],
    out_pdf_dir: str | Path,
    session: PoliteSession,
    *,
    limit: int = -1,
) -> int:
    """Download each record's PDF into ``out_pdf_dir/<safe-identifier>.pdf``; return the count.

    Skips records without an ``article_url`` and records whose declared language is non-Turkish
    (see :func:`_is_turkish`). For each kept record the article landing page is fetched and its
    full-text PDF URL resolved (:func:`resolve_pdf_url`); a record whose page yields no PDF link
    is logged and skipped. The resolved URL is validated with :func:`_is_safe_pdf_url` (SSRF
    guard: scheme + private-host + allowlist) before any fetch. Filenames are sanitised against
    path traversal (:func:`_safe_filename`). ``limit`` (``-1`` = all) caps *successful* downloads
    so a smoke run stays small. The PDF is streamed and aborted if it exceeds
    :data:`MAX_PDF_BYTES`; a single failed/oversized/unsafe download is logged and skipped so one
    bad article never aborts the batch.
    """
    out = Path(out_pdf_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for record in records:
        if 0 < limit <= downloaded:
            return downloaded
        article_url = record.get("article_url")
        if not article_url:
            continue
        if not _is_turkish(record):
            continue
        pdf_url = resolve_pdf_url(session, article_url)
        if not pdf_url:
            logger.warning("no PDF link on article page, skipping: %s", article_url)
            continue
        if not _is_safe_pdf_url(pdf_url):
            logger.warning("skipping unsafe pdf_url (SSRF guard): %s", pdf_url)
            continue
        content = _download_pdf_bytes(session, pdf_url)
        if content is None:
            continue
        dest = out / _safe_filename(record.get("identifier") or "record")
        dest.write_bytes(content)
        downloaded += 1
    return downloaded


def _download_pdf_bytes(session: PoliteSession, pdf_url: str) -> bytes | None:
    """Stream a PDF, returning its bytes, or ``None`` if the fetch fails or exceeds the cap.

    Streams in 64 KiB chunks and aborts as soon as the accumulated size would exceed
    :data:`MAX_PDF_BYTES`, so a hostile/broken endpoint can't make us buffer an unbounded body.
    Any network error is logged and yields ``None`` so one bad article never aborts the batch.
    """
    try:
        resp = session.get(pdf_url, stream=True)
    except Exception as exc:  # noqa: BLE001 — one bad fetch must not abort the harvest
        logger.warning("failed to download %s: %s", pdf_url, exc)
        return None
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in resp.iter_content(65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_PDF_BYTES:
                logger.warning("PDF exceeds %d bytes, skipping: %s", MAX_PDF_BYTES, pdf_url)
                return None
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001 — a stream error mid-download: skip this article
        logger.warning("failed while streaming %s: %s", pdf_url, exc)
        return None
    return b"".join(chunks)


def download_dergipark(
    out_pdf_dir: str | Path,
    *,
    set_spec: str | None = None,
    max_records: int = -1,
    user_agent: str = DEFAULT_USER_AGENT,
    min_delay: float = 1.0,
) -> int:
    """Harvest DergiPark over OAI-PMH and download article PDFs into ``out_pdf_dir``; count.

    Convenience wrapper: builds a :class:`PoliteSession` with ``respect_robots=False`` (OAI-PMH
    is the sanctioned harvest channel, but we still throttle + back off), harvests records, and
    downloads their PDFs. ``max_records`` caps the harvest; ``set_spec`` restricts to one OAI
    set (per-journal/publisher — VERIFY LIVE).

    After this, extract the PDFs to JSONL with the academic ingester::

        uv run --extra academic python scripts/ingest_academic.py \\
            --source dergipark --pdf-dir <out_pdf_dir>

    then clean the JSONL through the existing pipeline (``tc-run-hplt --source jsonl``).
    DergiPark articles are mostly CC BY per-journal — verify each journal's license.
    """
    session = PoliteSession(
        user_agent=user_agent,
        min_delay=min_delay,
        respect_robots=False,  # OAI-PMH is designed for harvesting (see module docstring)
    )
    records = harvest_records(session, set_spec=set_spec, max_records=max_records)
    return download_pdfs(records, out_pdf_dir, session, limit=max_records)
