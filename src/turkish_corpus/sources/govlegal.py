"""Turkish government / legal source ingesters (roadmap step 2, ``legal`` register).

Turkish legislation, the daily official gazette, court decisions and parliamentary
transcripts are *official texts* — published by the state without a private copyright
holder, so they are freely reusable. They also supply the formal, high-register Turkish
that web crawl alone underweights, which is why the blend wants them.

Each ingester emits **raw** datatrove-ready ``{"id","text","metadata"}`` records via the
shared contract (:mod:`.base`); the EXISTING pipeline (``tc-run-hplt --source jsonl``) does
all normalization, KVKK PII scrubbing, and dedup — we never reimplement those here.

Only :func:`ingest_mevzuat` is live: it streams a Hugging Face mirror of mevzuat.gov.tr, so
no scraping is required. The other three (Resmî Gazete, courts, TBMM) need polite scrapers
and are documented :exc:`NotImplementedError` SCAFFOLDS for a later step — each docstring
records the portal, the politeness obligation, the license, and the rough token volume so
the next implementer can pick them up cold.

Heavy deps (``datasets``) import lazily so this module imports without the ``sources`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import SourceInfo, make_record, write_records

__all__ = [
    "MEVZUAT",
    "RESMI_GAZETE",
    "COURT_DECISIONS",
    "TBMM_TUTANAK",
    "ingest_mevzuat",
    "ingest_resmi_gazete",
    "ingest_court_decisions",
    "ingest_tbmm_tutanak",
]

# --- Source provenance ----------------------------------------------------------------
# All four are official state texts: public, no private copyright holder. The license
# string is carried into every record's metadata so the blend manifest can audit it.
MEVZUAT = SourceInfo(
    name="mevzuat",
    license="public (official text)",
    register="legal",
    description="Turkish legislation from mevzuat.gov.tr (HF mirror)",
)
RESMI_GAZETE = SourceInfo(
    name="resmi_gazete",
    license="public (official text)",
    register="legal",
    description="Resmî Gazete, the daily Turkish official gazette (resmigazete.gov.tr)",
)
COURT_DECISIONS = SourceInfo(
    name="court_decisions",
    license="public (official text)",
    register="legal",
    description="Turkish high-court decisions (Yargıtay / Danıştay)",
)
TBMM_TUTANAK = SourceInfo(
    name="tbmm_tutanak",
    license="public (official text)",
    register="legal",
    description="TBMM (Grand National Assembly) plenary transcripts",
)

# Hugging Face mirror of mevzuat.gov.tr. The exact text field name is verified at runtime
# (see _record_from_row); we never hit the network in tests or at import.
_MEVZUAT_HF_DATASET = "muhammetakkurt/mevzuat-gov-dataset"

# Text keys to try, in priority order. The dataset's column is confirmed at runtime, so we
# probe the common shapes a mevzuat/HF text dataset uses rather than hardcode one name.
_TEXT_KEYS = ("text", "content", "madde", "icerik")


def _record_from_row(row: dict, idx: int) -> dict | None:
    """Build one ``make_record`` dict from a streamed dataset row, or ``None`` if textless.

    The dataset's exact text column is verified at runtime — we don't have network access to
    inspect it offline — so we try the common Turkish/HF text keys in priority order
    (``text`` → ``content`` → ``madde`` → ``icerik``) and take the first non-empty value.
    The id comes from ``row["id"]`` when present, else the streaming index, so every record
    has a stable, unique id even for a column-less dataset. Rows with no usable text return
    ``None`` and are skipped by the caller (downstream filters would drop them anyway).
    """
    text = None
    for key in _TEXT_KEYS:
        value = row.get(key)
        if value:
            text = value
            break
    if not text:
        return None
    doc_id = f"mevzuat-{row.get('id', idx)}"
    return make_record(str(text), doc_id, MEVZUAT)


def _records(rows: Iterable[dict], limit: int) -> Iterator[dict]:
    """Yield records from ``rows``, stopping after ``limit`` emitted (``limit < 0`` = all).

    A generator (not a list) so callers stream straight into :func:`write_records` without
    materializing the whole dataset, and so tests can drive it with a small fake iterable.
    The cap counts *emitted* records, not rows consumed, so textless rows don't eat budget.
    """
    emitted = 0
    for idx, row in enumerate(rows):
        if 0 <= limit <= emitted:
            return
        rec = _record_from_row(row, idx)
        if rec is None:
            continue
        yield rec
        emitted += 1


def ingest_mevzuat(out_dir: str, *, limit: int = -1) -> int:
    """Stream the mevzuat.gov.tr HF mirror into raw JSONL under ``out_dir``; return count.

    Streams (``streaming=True``) so the dataset is never fully downloaded — it flows row by
    row into the gzipped shard. ``limit`` caps emitted records for quick smoke tests
    (``-1`` = the whole split). ``datasets`` is imported lazily so the module loads without
    the ``sources`` extra; only this call needs it.
    """
    from datasets import load_dataset  # noqa: PLC0415

    rows = load_dataset(_MEVZUAT_HF_DATASET, split="train", streaming=True)
    return write_records(_records(rows, limit), out_dir)


def ingest_resmi_gazete(out_dir: str, *, limit: int = -1) -> int:
    """SCAFFOLD — Resmî Gazete (daily Turkish official gazette). Not yet implemented.

    Portal: https://www.resmigazete.gov.tr — the official gazette publishes every law,
    regulation, and appointment daily, with a dated archive reaching back decades. The
    daily editions are linked from the date-indexed archive and served as HTML (older
    issues as PDF), so this needs a polite, dated scraper rather than an HF download.

    Approach: walk the date archive, fetch each day's HTML (PDF text extraction for the
    pre-digital back-catalogue), and emit one record per notice via :func:`make_record`
    with :data:`RESMI_GAZETE`. Reuse the crawl package's politeness patterns
    (``turkish_corpus.crawl.settings``: ``ROBOTSTXT_OBEY``, AutoThrottle, one request per
    domain, 429/503 retry) — do not hammer a single government host.

    License: public (official text). Rough volume: the gazette is modest next to the courts
    (well under 1B tokens) but high-value for legal/administrative register coverage.
    """
    raise NotImplementedError(
        "ingest_resmi_gazete is a scaffold: scrape resmigazete.gov.tr's dated archive with "
        "the crawl package's polite settings (robots, AutoThrottle, 1 req/domain) and emit "
        "records via make_record(..., RESMI_GAZETE). See this function's docstring."
    )


def ingest_court_decisions(out_dir: str, *, limit: int = -1) -> int:
    """SCAFFOLD — Turkish high-court decisions (Yargıtay / Danıştay). Not yet implemented.

    Portals: Yargıtay (Court of Cassation) https://karararama.yargitay.gov.tr and Danıştay
    (Council of State) https://www.danistay.gov.tr karar-arama. Both expose searchable
    decision databases behind a query/pagination API returning decision text per ruling.

    Approach: page through the search API per chamber/date window, fetch each decision's
    full text, and emit one record per decision via :func:`make_record` with
    :data:`COURT_DECISIONS` (stamp chamber/date/esas-karar numbers into metadata for
    provenance). Reuse the crawl package's politeness patterns
    (``turkish_corpus.crawl.settings``) — these are state hosts with rate limits; obey
    robots, throttle to one request per domain, and back off on 429/503.

    License: public (official text). Rough volume: court decisions are the largest legal
    source by far — research puts them at ~3.4B tokens — so this is the highest-yield
    scaffold here and likely needs sharding rather than a single output file.
    """
    raise NotImplementedError(
        "ingest_court_decisions is a scaffold: page Yargıtay/Danıştay decision-search APIs "
        "politely (crawl settings: robots, AutoThrottle, 1 req/domain, 429/503 backoff) and "
        "emit records via make_record(..., COURT_DECISIONS). ~3.4B tokens; shard the output. "
        "See this function's docstring."
    )


def ingest_tbmm_tutanak(out_dir: str, *, limit: int = -1) -> int:
    """SCAFFOLD — TBMM parliamentary transcripts (tutanak). Not yet implemented.

    Portal: https://www.tbmm.gov.tr — the Grand National Assembly publishes verbatim
    plenary transcripts (genel kurul tutanakları) per legislative term and session, mostly
    as HTML pages and downloadable documents indexed by term/session.

    Approach: enumerate the term/session index, fetch each session's transcript (HTML, with
    document/PDF text extraction where needed), and emit one record per session (or per
    speech turn) via :func:`make_record` with :data:`TBMM_TUTANAK`. Reuse the crawl
    package's politeness patterns (``turkish_corpus.crawl.settings``): obey robots,
    AutoThrottle, one request per domain, retry on 429/503.

    License: public (official text). Rough volume: substantial spoken-but-formal Turkish
    spanning many terms — valuable register diversity, smaller than the court corpus.
    """
    raise NotImplementedError(
        "ingest_tbmm_tutanak is a scaffold: walk TBMM's term/session transcript index "
        "politely (crawl settings: robots, AutoThrottle, 1 req/domain, 429/503 backoff) and "
        "emit records via make_record(..., TBMM_TUTANAK). See this function's docstring."
    )
