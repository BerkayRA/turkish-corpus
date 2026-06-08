"""Fetch a single WARC record from Common Crawl by HTTP byte range.

A spot-check / verification helper, not part of the bulk crawl path. The CC index rows from
:mod:`turkish_corpus.crawl.cc_index` carry ``warc_filename``, ``warc_record_offset`` and
``warc_record_length``; with those you can pull *one* page's stored HTML straight out of
CC's WARC files via an HTTP ``Range`` request — no full-file download — to confirm the
index points at real Turkish HTML before committing to a crawl (this is the cheap
"UnifiedCrawl"-style targeted fetch the roadmap mentions).

``requests`` and ``warcio`` are imported lazily (``crawl`` extra) so the module imports
without them; the function returns ``None`` on any non-record / decode problem rather than
raising, because it is a best-effort diagnostic.
"""

from __future__ import annotations

import pathlib

__all__ = ["fetch_warc_record", "CC_DATA_BASE_URL"]

# CC serves WARCs over plain HTTPS here (no S3 credentials needed); warc_filename is the
# bucket-relative path stored in the index.
CC_DATA_BASE_URL = "https://data.commoncrawl.org/"

# (connect, read) timeouts: a single int applies only to connect+first-byte and lets a
# server trickle bytes forever, so cap the read leg explicitly.
_REQUEST_TIMEOUT_SECS = (10, 20)

# A real WARC ``response`` record is well under 1 MB; cap the body we will buffer so an
# inflated/adversarial ``length`` from the index can't OOM the process.
_MAX_RECORD_BYTES = 50 * 1024 * 1024


def _safe_warc_path(warc_filename: str) -> str | None:
    """Reject anything that isn't a plain bucket-relative path (URL / traversal)."""
    cleaned = warc_filename.lstrip("/")
    parsed = pathlib.PurePosixPath(cleaned)
    if "://" in warc_filename or ".." in parsed.parts:
        return None
    return cleaned


def fetch_warc_record(warc_filename: str, offset: int, length: int) -> str | None:
    """Return the decoded HTML of one CC WARC ``response`` record, or ``None``.

    Issues ``GET https://data.commoncrawl.org/<warc_filename>`` with a
    ``Range: bytes=<offset>-<offset+length-1>`` header (CC answers ``206 Partial Content``),
    parses the returned bytes with ``warcio``, takes the first ``response`` record, and
    decodes its payload as UTF-8 (replacing undecodable bytes).

    Parameters
    ----------
    warc_filename:
        Bucket-relative WARC path from the index (``warc_filename`` column).
    offset:
        Byte offset of the record (``warc_record_offset`` column).
    length:
        Record length in bytes (``warc_record_length`` column).

    Returns ``None`` if the request fails, the range is unexpected, or no ``response``
    record is found — callers treat that as "could not verify".
    """
    import io  # noqa: PLC0415

    import requests  # noqa: PLC0415  (crawl extra; lazy)
    from warcio.archiveiterator import ArchiveIterator  # noqa: PLC0415

    offset, length = int(offset), int(length)
    if offset < 0 or length <= 0 or length > _MAX_RECORD_BYTES:
        return None
    cleaned = _safe_warc_path(warc_filename)
    if cleaned is None:
        return None

    url = CC_DATA_BASE_URL + cleaned
    end = offset + length - 1
    headers = {"Range": f"bytes={offset}-{end}"}

    try:
        resp = requests.get(
            url, headers=headers, timeout=_REQUEST_TIMEOUT_SECS, stream=True
        )
    except requests.RequestException:
        return None
    try:
        # 206 is the expected partial-content answer; bail on anything else (incl. 200 = the
        # server ignored the range and would stream the whole multi-GB file).
        if resp.status_code != 206:
            return None
        # Read at most one byte past the cap so we can detect (and reject) an overlong body
        # without buffering it all.
        body = resp.raw.read(_MAX_RECORD_BYTES + 1, decode_content=True)
    finally:
        resp.close()
    if len(body) > _MAX_RECORD_BYTES:
        return None

    for record in ArchiveIterator(io.BytesIO(body)):
        if record.rec_type == "response":
            payload = record.content_stream().read()
            return payload.decode("utf-8", errors="replace")
    return None
