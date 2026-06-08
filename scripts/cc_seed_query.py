"""CLI: rank the highest-yield Turkish hosts in a Common Crawl snapshot (roadmap 3a).

Runs a DuckDB query over Common Crawl's public columnar Parquet index and writes a ranked
``url_host_name,pages`` seed CSV for the Scrapy crawler (``scripts/run_crawl.py``). No
credentials needed (anonymous access to the public CC bucket).

Examples
--------
Default recent snapshot, primary-Turkish hosts with >=100 pages, top 5000::

    uv run --extra crawl python scripts/cc_seed_query.py \\
        --out tr_hosts.csv --min-pages 100

Prototype against a few index files (fast/cheap), including secondary-Turkish pages::

    uv run --extra crawl python scripts/cc_seed_query.py \\
        --crawl-id CC-MAIN-2025-51 --parts 'part-00000-*.parquet' \\
        --all-turkish --limit 200 --out tr_hosts_sample.csv

Verify the latest crawl id at https://data.commoncrawl.org before a real run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.crawl.cc_index import DEFAULT_CRAWL_ID, run_host_query  # noqa: E402


def _crawl_id(value: str) -> str:
    """argparse type: reject malformed crawl ids early with a friendly message.

    Defence-in-depth; cc_index also validates before interpolating into SQL.
    """
    if not re.fullmatch(r"CC-MAIN-\d{4}-\d{1,2}", value):
        raise argparse.ArgumentTypeError(
            f"invalid crawl id {value!r}; expected 'CC-MAIN-YYYY-NN'"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cc_seed_query",
        description="Rank Turkish hosts in a Common Crawl snapshot into a seed CSV.",
    )
    p.add_argument("--crawl-id", default=DEFAULT_CRAWL_ID, type=_crawl_id,
                   help=f"CC monthly snapshot id (default: {DEFAULT_CRAWL_ID}). Verify latest "
                        "at data.commoncrawl.org.")
    p.add_argument("--out", default="tr_hosts.csv", help="Output CSV path.")
    p.add_argument("--min-pages", type=int, default=50,
                   help="Minimum Turkish HTML pages per host (default: 50).")
    p.add_argument("--limit", type=int, default=5000,
                   help="Max hosts to return, ranked by page count (default: 5000).")
    p.add_argument("--parts", default=None,
                   help="Index file glob within the partition (e.g. 'part-00000-*.parquet') "
                        "for cheap prototyping; default scans all '*.parquet'.")
    p.add_argument("--all-turkish", action="store_true",
                   help="Include hosts where Turkish is only a secondary language "
                        "(content_languages LIKE '%%tur%%'), not just primary.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Querying CC index {args.crawl_id} (this scans remote Parquet; may take a while)...")
    rows = run_host_query(
        args.crawl_id,
        args.out,
        min_pages=args.min_pages,
        primary_only=not args.all_turkish,
        limit=args.limit,
        parts=args.parts,
    )
    print(f"Wrote {rows} hosts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
