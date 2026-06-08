"""Academic source ingesters — YÖKTEZ theses + DergiPark articles (roadmap 2, ``academic``).

Theses (YÖK Ulusal Tez Merkezi) and open-access journal articles (DergiPark) supply the
long-form, formal, terminology-dense Turkish that web crawl underweights — the register the
blend wants from academia. Both arrive as **PDFs**, so this module's real work is turning a
directory of downloaded PDFs into raw datatrove-ready ``{"id","text","metadata"}`` records
via the shared contract (:mod:`.base`); the EXISTING pipeline (``tc-run-hplt --source jsonl``)
does all normalization, KVKK PII scrubbing, and dedup — never reimplemented here.

Download is deliberately split from extraction. Fetching the PDFs is the hard, polite,
portal-specific part (OAI-PMH for DergiPark; a session/CAPTCHA-gated portal for YÖKTEZ), so
:func:`download_dergipark` / :func:`download_yoktez` are documented ``NotImplementedError``
SCAFFOLDS. Once PDFs sit on disk, :func:`ingest_pdfs` extracts them with no network at all.

OCR NOTE
--------
:func:`extract_pdf_text` only reads a PDF's *text layer* (born-digital PDFs). SCANNED theses
— image-only PDFs with no text layer — yield nothing and are reported as *needing OCR*. An
OCR pass (tesseract via ``ocrmypdf`` or ``pytesseract``, with Turkish ``tur`` language data)
would recover them, but OCR is heavy and noisy, so it is intentionally NOT a dependency of
this module: run OCR upstream to add a text layer, then re-run :func:`ingest_pdfs`.

Heavy deps (``pypdf``, ``pdfplumber``) import lazily so this module imports without the
``academic`` extra; only :func:`extract_pdf_text` pulls them in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from .base import SourceInfo, make_record, write_records

__all__ = [
    "DERGIPARK",
    "YOKTEZ",
    "extract_pdf_text",
    "ingest_pdfs",
    "ingest_dergipark",
    "ingest_yoktez",
    "download_dergipark",
    "download_yoktez",
]

logger = logging.getLogger(__name__)

# An extractor takes a PDF path and returns its text, or None if the text layer is too thin
# (likely scanned). Aliased so :func:`ingest_pdfs` / :func:`_records_from_pdfs` can accept an
# injectable extractor — tests stub this to drive the pipeline without real PDFs.
PdfExtractor = Callable[[Path], "str | None"]

# Below this many extracted characters we treat a PDF as scanned (text layer absent/garbage)
# and hand it to OCR instead of emitting a near-empty record. 200 chars ≈ a short paragraph:
# enough to clear cover-page noise while not discarding genuinely short born-digital docs.
_DEFAULT_MIN_CHARS = 200

# --- Source provenance ----------------------------------------------------------------
# License strings are carried into every record's metadata so the blend manifest can audit
# them. They are conservative on purpose: DergiPark licensing is per-journal (mostly CC BY
# but verify each), and theses are author-permitted for research rather than blanket-open.
DERGIPARK = SourceInfo(
    name="dergipark",
    license="CC BY (per-journal; verify)",
    register="academic",
    description="Open-access Turkish journal articles from DergiPark",
)
YOKTEZ = SourceInfo(
    name="yoktez",
    license="author-permitted / research",
    register="academic",
    description="Turkish master's/PhD theses from YÖK Ulusal Tez Merkezi",
)


def extract_pdf_text(pdf_path: str | Path, *, min_chars: int = _DEFAULT_MIN_CHARS) -> str | None:
    """Extract the text layer of a born-digital PDF; ``None`` if it's likely scanned.

    Strategy (two libraries, escalating): ``pypdf`` first because it is fast and handles most
    born-digital PDFs; if its output is under ``min_chars`` we fall back to ``pdfplumber``,
    which has a stronger layout engine and often recovers text ``pypdf`` misses. If *both*
    stay under ``min_chars`` the PDF almost certainly has no usable text layer (it's scanned),
    so we return ``None`` to signal "needs OCR" rather than emit a near-empty record.

    Robustness is the priority in a batch: a single malformed PDF or a library blowing up on
    one page must never crash the whole ingest. Every page-level and library-level extraction
    is wrapped so failures degrade to "less text" (and ultimately ``None``), never an
    exception escaping to the caller.

    ``pypdf`` / ``pdfplumber`` are imported lazily so the module loads without the
    ``academic`` extra; only this function needs them.
    """
    path = Path(pdf_path)

    text = _extract_with_pypdf(path).strip()
    if len(text) >= min_chars:
        return text

    # pypdf came up short — try pdfplumber's stronger layout extraction before giving up.
    plumber_text = _extract_with_pdfplumber(path).strip()
    if len(plumber_text) >= len(text):
        text = plumber_text

    if len(text) < min_chars:
        # No usable text layer from either library → scanned PDF, needs an OCR pass.
        return None
    return text


def _extract_with_pypdf(path: Path) -> str:
    """Return concatenated page text via ``pypdf``, or ``""`` on any failure.

    Per-page extraction is isolated so one unreadable page (a common corruption) loses just
    that page's text instead of the whole document; opening the file failing returns ``""``
    so the caller can fall through to ``pdfplumber``.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError:
        # Misconfiguration (academic extra not installed): treat as "no text" so the caller
        # can still try pdfplumber, but it's worth a warning to the operator.
        logger.warning("pypdf not installed; install the 'academic' extra to extract PDF text")
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 — any malformed/encrypted PDF; degrade, don't crash
        logger.debug("pypdf could not open %s: %s", path.name, exc)
        return ""

    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 — one bad page must not lose the rest
            logger.debug("pypdf failed on a page of %s: %s", path.name, exc)
    return "\n".join(parts)


def _extract_with_pdfplumber(path: Path) -> str:
    """Return concatenated page text via ``pdfplumber``, or ``""`` on any failure.

    Same per-page isolation as :func:`_extract_with_pypdf`; pdfplumber is the heavier, more
    layout-aware fallback used only when pypdf came up short.
    """
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "pdfplumber not installed; install the 'academic' extra for the extraction fallback"
        )
        return ""

    try:
        with pdfplumber.open(str(path)) as pdf:
            parts: list[str] = []
            for page in pdf.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception as exc:  # noqa: BLE001 — one bad page must not lose the rest
                    logger.debug("pdfplumber failed on a page of %s: %s", path.name, exc)
            return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001 — malformed/encrypted PDF; degrade, don't crash
        logger.debug("pdfplumber could not open %s: %s", path.name, exc)
        return ""


class _PdfRecords:
    """A lazy, single-pass record iterator that also tallies PDFs needing OCR.

    Takes an injectable ``extractor`` (signature :data:`PdfExtractor`) so tests can stub
    extraction and exercise the whole pipeline without real PDFs. It is an *iterator* (not a
    list) so PDFs stream straight into :func:`write_records` without materializing every text
    in memory — theses are large.

    A plain generator can't carry the OCR tally (generators reject attribute assignment), so
    a tiny iterator class exposes :attr:`needs_ocr` — read it *after* iteration completes.
    Paths the extractor returns falsy/``None`` for are *scanned* (need OCR): skipped but
    tallied. ``limit`` (``<= 0`` = all) counts *emitted* records, so scanned PDFs don't eat
    the budget.
    """

    def __init__(
        self, paths: Iterable[Path], source: SourceInfo, extractor: PdfExtractor, limit: int
    ) -> None:
        self._paths = paths
        self._source = source
        self._extractor = extractor
        self._limit = limit
        self.needs_ocr = 0

    def __iter__(self) -> Iterator[dict]:
        emitted = 0
        for path in self._paths:
            if 0 < self._limit <= emitted:
                return
            try:
                text = self._extractor(path)
            except Exception as exc:  # noqa: BLE001 — a bad extractor must not crash the batch
                logger.warning("extraction raised on %s: %s", path.name, exc)
                text = None
            if not text:
                # Scanned (or unreadable): no text layer. Tally for the OCR backlog, skip it.
                self.needs_ocr += 1
                continue
            doc_id = f"{self._source.name}-{path.stem}"
            yield make_record(text, doc_id, self._source, filename=path.name)
            emitted += 1


def _records_from_pdfs(
    paths: Iterable[Path],
    source: SourceInfo,
    extractor: PdfExtractor,
    limit: int,
) -> _PdfRecords:
    """Build a :class:`_PdfRecords` iterator over ``paths`` (see that class for semantics)."""
    return _PdfRecords(paths, source, extractor, limit)


def ingest_pdfs(
    pdf_dir: str | Path,
    out_dir: str,
    source: SourceInfo,
    *,
    limit: int = -1,
    glob: str = "*.pdf",
) -> int:
    """Extract every PDF under ``pdf_dir`` into raw JSONL in ``out_dir``; return count written.

    Walks ``pdf_dir`` for ``glob`` (sorted for deterministic, resumable ordering), extracts
    each via :func:`extract_pdf_text`, and writes one record per born-digital PDF. PDFs with
    no usable text layer (scanned) are skipped and logged as needing OCR — the written count
    reflects only what was emitted, and the OCR backlog is logged at INFO so an operator can
    see how many theses still need an OCR pass. No network: extraction is fully local.

    ``limit`` (``-1`` = all) caps emitted records for quick smoke tests.
    """
    paths = sorted(Path(pdf_dir).glob(glob))
    records = _records_from_pdfs(paths, source, extract_pdf_text, limit)
    written = write_records(records, out_dir)
    needs_ocr = getattr(records, "needs_ocr", 0)
    logger.info(
        "[%s] wrote %d record(s) to %s; %d PDF(s) skipped as scanned (need OCR)",
        source.name,
        written,
        out_dir,
        needs_ocr,
    )
    return written


def ingest_dergipark(out_dir: str, *, pdf_dir: str | Path, limit: int = -1) -> int:
    """Ingest already-downloaded DergiPark article PDFs into raw JSONL; return count written.

    Thin wrapper over :func:`ingest_pdfs` with :data:`DERGIPARK`. PDFs must be downloaded
    first (see :func:`download_dergipark`); this step is offline extraction only.
    """
    return ingest_pdfs(pdf_dir, out_dir, DERGIPARK, limit=limit)


def ingest_yoktez(out_dir: str, *, pdf_dir: str | Path, limit: int = -1) -> int:
    """Ingest already-downloaded YÖKTEZ thesis PDFs into raw JSONL; return count written.

    Thin wrapper over :func:`ingest_pdfs` with :data:`YOKTEZ`. PDFs must be downloaded first
    (see :func:`download_yoktez`); this step is offline extraction only.
    """
    return ingest_pdfs(pdf_dir, out_dir, YOKTEZ, limit=limit)


def download_dergipark(out_pdf_dir: str | Path, *, limit: int = -1) -> int:
    """SCAFFOLD — download open-access DergiPark article PDFs. Not yet implemented.

    Portal: https://dergipark.org.tr — DergiPark hosts thousands of Turkish open-access
    journals and, helpfully, exposes an **OAI-PMH** endpoint (``/oai``) for harvesting
    article metadata (Dublin Core: title, authors, identifiers, and the per-article landing
    URL). The full-text PDF for each article hangs off the article page under a stable
    ``.../download/article-file/<id>`` style URL.

    Approach: harvest records over OAI-PMH (page with ``resumptionToken``), resolve each
    article's PDF URL, and download PDFs into ``out_pdf_dir`` (named by a stable id so
    :func:`ingest_dergipark` produces deterministic doc ids). Be polite: obey robots, set a
    descriptive User-Agent, throttle to roughly one request per few seconds, and back off on
    429/503 — reuse the crawl package's politeness settings
    (``turkish_corpus.crawl.settings``). Record each journal's license from the OAI metadata
    so the per-journal CC terms can be verified rather than assumed.

    Then run :func:`ingest_dergipark` to extract the downloaded PDFs.
    """
    raise NotImplementedError(
        "download_dergipark is a scaffold: harvest DergiPark over OAI-PMH (/oai, paginate "
        "with resumptionToken), resolve per-article PDF URLs, and download politely (robots, "
        "User-Agent, throttle, 429/503 backoff) into out_pdf_dir, then run ingest_dergipark. "
        "See this function's docstring."
    )


def download_yoktez(out_pdf_dir: str | Path, *, limit: int = -1) -> int:
    """SCAFFOLD — download YÖKTEZ thesis PDFs. Not yet implemented (download is the hard part).

    Portal: https://tez.yok.gov.tr — YÖK's Ulusal Tez Merkezi (National Thesis Center). The
    catalogue is searchable, but downloading the full-text PDFs is the genuinely hard part:
    the portal is **session-gated and CAPTCHA-protected**, theses can be access-restricted by
    the author for an embargo period, and bulk automated download is discouraged. There is no
    open API or bulk dump — expect to drive an authenticated session and solve/avoid CAPTCHAs,
    and to honour per-thesis access permissions (author-permitted / research use only).

    Approach: obtain proper access, drive a session that respects the portal's terms, fetch
    only permitted full-text PDFs into ``out_pdf_dir`` (named by thesis number for stable doc
    ids), and be extremely polite — minimal concurrency, generous delays, robots, and backoff
    on throttling (reuse ``turkish_corpus.crawl.settings``). Many theses are SCANNED, so plan
    for an OCR pass before/after extraction (see this module's OCR note).

    Then run :func:`ingest_yoktez` to extract the downloaded PDFs.
    """
    raise NotImplementedError(
        "download_yoktez is a scaffold: the YÖK portal (tez.yok.gov.tr) is session/CAPTCHA-"
        "gated with per-thesis access permissions, so download is the hard part — obtain "
        "access, drive a polite authenticated session (robots, throttle, 429/503 backoff), "
        "honour author permissions, then run ingest_yoktez. Many theses are scanned and need "
        "OCR. See this function's docstring."
    )
