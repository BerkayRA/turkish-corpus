"""Turkish legislation ingester — the mevzuat.gov.tr Hugging Face mirror (``legal`` register).

Turkish legislation is *official text* — published by the state with no private copyright
holder, so it is freely reusable — and supplies the formal, high-register Turkish that web
crawl underweights. This module covers the one legal source available as a ready HF dataset
(no scraping): a mirror of mevzuat.gov.tr.

The other legal sources (Resmî Gazete, Yargıtay/Danıştay court decisions, TBMM transcripts)
need polite live scrapers and live in :mod:`turkish_corpus.sources.govscrape`
(``scripts/download_govlegal.py``) — Resmî Gazete and TBMM are implemented there; courts and
YÖKTEZ are documented scaffolds (JS/CAPTCHA-gated).

Records are emitted via the shared contract (:mod:`.base`); the EXISTING pipeline
(``tc-run-hplt --source jsonl``) does all normalization, KVKK PII scrubbing, and dedup.
``datasets`` imports lazily so this module loads without the ``sources`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import SourceInfo, make_record, write_records

__all__ = ["MEVZUAT", "ingest_mevzuat"]

# Official state text: public, no private copyright holder. Carried into every record's
# metadata so the blend manifest can audit licensing.
MEVZUAT = SourceInfo(
    name="mevzuat",
    license="public (official text)",
    register="legal",
    description="Turkish legislation from mevzuat.gov.tr (HF mirror)",
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
    """Yield records from ``rows``, stopping after ``limit`` emitted (``limit <= 0`` = all).

    A generator (not a list) so callers stream straight into :func:`write_records` without
    materializing the whole dataset, and so tests can drive it with a small fake iterable.
    The cap counts *emitted* records, not rows consumed, so textless rows don't eat budget.
    ``limit=0`` means "no limit" (consistent with the other ingesters), not "yield nothing".
    """
    emitted = 0
    for idx, row in enumerate(rows):
        if 0 < limit <= emitted:
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
