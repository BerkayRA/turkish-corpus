"""Query Common Crawl's columnar (Parquet) index for the highest-yield Turkish hosts.

This is roadmap step **3a**. Rather than crawl the live web first, we mine Common Crawl's
public, Hive-partitioned Parquet index — it already tells us, per host, how many Turkish
``text/html`` pages CC successfully fetched. Ranking hosts by that count gives an
empirical, traffic-weighted ``.tr`` seed list for the Scrapy crawler (3b) without a single
live request, which is cheaper, faster, and lower legal risk (the roadmap's verdict: mine
CC before crawling fresh).

Design split (mirrors the rest of the package):

- :func:`build_index_path` / :func:`build_tr_host_query` are **pure** — they only build
  strings (an S3 glob and a DuckDB SQL statement). They import nothing heavy and are fully
  unit-testable, so the exact SQL we send to a public dataset is asserted in CI.
- :func:`run_host_query` is the only function that touches DuckDB + httpfs (network). It
  imports ``duckdb`` lazily so ``import turkish_corpus.crawl.cc_index`` stays cheap and the
  pure core remains importable without the ``crawl`` extra.

Why ``content_languages LIKE 'tur%'`` and not ``= 'tur'``: the index stores ISO-639-3
codes (Turkish is ``tur``, *not* ``tr``) comma-separated in *confidence order*. A page
whose dominant language is Turkish looks like ``tur`` or ``tur,eng`` — i.e. it *starts*
with ``tur``. ``LIKE 'tur%'`` keeps primary-Turkish pages; ``LIKE '%tur%'`` (primary_only
disabled) also keeps pages where Turkish is a secondary language, at the cost of more
multilingual noise.
"""

from __future__ import annotations

import re

__all__ = [
    "CC_INDEX_PATH_TEMPLATE",
    "DEFAULT_CRAWL_ID",
    "build_index_path",
    "build_tr_host_query",
    "run_host_query",
]

# crawl_id and parts are interpolated into the SQL string sent to DuckDB, so they must be
# validated to a strict shape before interpolation — never trust them as free text even
# though they normally come from a CLI. crawl ids look like ``CC-MAIN-2025-51``; ``parts``
# is a Parquet filename glob (word chars, dash, dot, star only).
_CRAWL_ID_RE = re.compile(r"^CC-MAIN-\d{4}-\d{1,2}$")
_PARTS_RE = re.compile(r"^[\w\-.*]+$")


def _validate_crawl_id(crawl_id: str) -> str:
    if not _CRAWL_ID_RE.fullmatch(crawl_id):
        raise ValueError(
            f"Invalid crawl_id {crawl_id!r}; expected the form 'CC-MAIN-YYYY-NN'."
        )
    return crawl_id


def _validate_parts(parts: str) -> str:
    if not _PARTS_RE.fullmatch(parts):
        raise ValueError(
            f"Invalid parts glob {parts!r}; only letters, digits, '_', '-', '.', '*' allowed."
        )
    return parts

# Public, requester-pays-free bucket in us-east-1. ``{crawl_id}`` is the monthly snapshot
# (e.g. ``CC-MAIN-2025-51``) and is the only Hive partition we vary; ``subset=warc`` keeps
# us on the WARC (page-fetch) records rather than robotstxt/crawldiagnostics subsets.
CC_INDEX_PATH_TEMPLATE = (
    "s3://commoncrawl/cc-index/table/cc-main/warc/crawl={crawl_id}/subset=warc/{parts}"
)

# A recent monthly snapshot. CC publishes a new crawl roughly monthly and old ones never
# move, so this default goes stale — verify/refresh the latest id at data.commoncrawl.org
# (see docs/crawler.md) and pass it via --crawl-id.
DEFAULT_CRAWL_ID = "CC-MAIN-2025-51"

# Drop the index below the length floor of a useful host list; a hard ceiling keeps a
# runaway query from materialising the entire long tail of one-page hosts.
_DEFAULT_LIMIT = 5000


def build_index_path(crawl_id: str, parts: str | None = None) -> str:
    """Build the S3 glob for one CC monthly index partition.

    Parameters
    ----------
    crawl_id:
        Monthly snapshot id, e.g. ``"CC-MAIN-2025-51"``.
    parts:
        File selector within the partition. ``None`` (default) scans every Parquet file
        (``*.parquet``) — the real run. For prototyping, pass a narrower glob such as
        ``"part-00000-*.parquet"`` to read a handful of files and keep the scan cheap.
    """
    crawl_id = _validate_crawl_id(crawl_id)
    parts = _validate_parts(parts) if parts else "*.parquet"
    return CC_INDEX_PATH_TEMPLATE.format(crawl_id=crawl_id, parts=parts)


def build_tr_host_query(
    crawl_id: str,
    *,
    min_pages: int = 50,
    primary_only: bool = True,
    limit: int = _DEFAULT_LIMIT,
    parts: str | None = None,
) -> str:
    """Return the DuckDB SQL that ranks Turkish hosts by fetched page count.

    Pure string builder — no DuckDB, no network. The query projects only the columns it
    needs (``url_host_name``) so DuckDB/httpfs can push the projection down and skip the
    rest of the very wide index, and filters on:

    - ``fetch_status = 200`` — only successfully fetched pages.
    - ``content_mime_detected = 'text/html'`` — HTML documents, not PDFs/images/feeds.
    - ``content_languages LIKE 'tur%'`` — primary-Turkish (``'%tur%'`` when
      ``primary_only`` is False, to also catch secondary-Turkish pages).

    ``HAVING pages >= :min_pages`` drops thin hosts; ``ORDER BY pages DESC LIMIT :limit``
    keeps the highest-yield head of the distribution.

    Parameters
    ----------
    crawl_id:
        Monthly snapshot id passed through to :func:`build_index_path`.
    min_pages:
        Minimum Turkish HTML pages a host must have to make the list.
    primary_only:
        Require Turkish to be the *dominant* language (``LIKE 'tur%'``). Disable to also
        include hosts where Turkish is only a secondary language (``LIKE '%tur%'``).
    limit:
        Maximum number of hosts to return (head of the ranked distribution).
    parts:
        Forwarded to :func:`build_index_path` (use a narrow glob for prototyping).
    """
    path = build_index_path(crawl_id, parts)
    lang_pattern = "tur%" if primary_only else "%tur%"
    # Built from validated literals/ints (crawl_id, min_pages, limit), not user-controlled
    # free text; kept as one readable statement so the SQL we run is auditable in tests.
    return (
        "SELECT url_host_name, COUNT(*) AS pages\n"
        f"FROM read_parquet('{path}')\n"
        "WHERE fetch_status = 200\n"
        "  AND content_mime_detected = 'text/html'\n"
        f"  AND content_languages LIKE '{lang_pattern}'\n"
        "GROUP BY url_host_name\n"
        f"HAVING pages >= {int(min_pages)}\n"
        "ORDER BY pages DESC\n"
        f"LIMIT {int(limit)}"
    )


def run_host_query(
    crawl_id: str = DEFAULT_CRAWL_ID,
    out_csv: str = "tr_hosts.csv",
    *,
    min_pages: int = 50,
    primary_only: bool = True,
    limit: int = _DEFAULT_LIMIT,
    parts: str | None = None,
) -> int:
    """Run the Turkish-host query against the public CC index and write a seed CSV.

    Lazily imports ``duckdb`` (the ``crawl`` extra) so module import stays cheap. Configures
    anonymous httpfs access to the public bucket (``s3_region='us-east-1'``; no credentials),
    runs :func:`build_tr_host_query`, and ``COPY``-s the ranked result to ``out_csv`` with a
    header (``url_host_name,pages``) so :mod:`turkish_corpus.crawl.seeds` can load it.

    Returns the number of host rows written.
    """
    import duckdb  # noqa: PLC0415  (crawl extra; lazy so the pure builders import freely)

    query = build_tr_host_query(
        crawl_id,
        min_pages=min_pages,
        primary_only=primary_only,
        limit=limit,
        parts=parts,
    )

    con = duckdb.connect()
    try:
        # httpfs reads the s3:// glob; anonymous access to the public CC bucket needs only
        # the region. No-sign-request is implied for the public bucket with no credentials.
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-east-1';")
        # COPY (<query>) streams straight to disk without materialising in Python.
        escaped = out_csv.replace("'", "''")
        con.execute(f"COPY ({query}) TO '{escaped}' (HEADER, DELIMITER ',')")
        # Re-count from the query for an accurate row total (COPY does not return one).
        (rows,) = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()
        return int(rows)
    finally:
        con.close()
