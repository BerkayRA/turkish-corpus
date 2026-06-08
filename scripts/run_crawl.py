"""CLI: launch the polite Turkish focused crawler (roadmap 3b).

Wraps a Scrapy ``CrawlerProcess`` configured from
:mod:`turkish_corpus.crawl.settings` and runs
:class:`~turkish_corpus.crawl.spider.TurkishSitemapSpider` over the hosts in a CC seed CSV,
writing one datatrove-ready JSONL record per page into ``--out``.

The JSONL it produces is read verbatim by the existing cleaning pipeline::

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/crawl_tur

Examples
--------
    uv run --extra crawl python scripts/run_crawl.py \\
        --hosts tr_hosts.csv --out output/crawl --min-pages 100

Equivalent to ``scrapy crawl turkish_sitemap -a hosts_csv=tr_hosts.csv``; this script just
sets the JSONL feed path and loads the project settings for you. Set a real USER_AGENT
contact in crawl/settings.py before any live run (see docs/crawler.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_crawl",
        description="Run the polite Turkish focused crawler over a CC seed host CSV.",
    )
    p.add_argument("--hosts", required=True,
                   help="CC seed CSV (url_host_name,pages) from scripts/cc_seed_query.py.")
    p.add_argument("--out", default="output/crawl",
                   help="Output directory for datatrove-ready JSONL (default: output/crawl).")
    p.add_argument("--min-pages", type=int, default=0,
                   help="Only crawl hosts with at least this many CC Turkish pages.")
    p.add_argument("--allow-placeholder-ua", action="store_true",
                   help="Override the safety check that blocks the placeholder USER_AGENT "
                        "contact (NOT for real crawls).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported here so --help works without the heavy crawl extra installed.
    from scrapy.crawler import CrawlerProcess  # noqa: PLC0415
    from scrapy.settings import Settings  # noqa: PLC0415

    from turkish_corpus.crawl import settings as crawl_settings  # noqa: PLC0415
    from turkish_corpus.crawl.spider import TurkishSitemapSpider  # noqa: PLC0415

    settings = Settings()
    settings.setmodule(crawl_settings, priority="project")

    # Refuse to crawl live hosts while still advertising the placeholder contact — site
    # operators must be able to reach a real human (RFC 9309 etiquette + KVKK posture).
    if not args.allow_placeholder_ua and "example.org" in settings.get("USER_AGENT", ""):
        raise SystemExit(
            "USER_AGENT still uses the placeholder contact (example.org). Set a real, "
            "monitored URL/email in turkish_corpus/crawl/settings.py before a live crawl "
            "(or pass --allow-placeholder-ua to override for local testing)."
        )

    # One JSONL feed per run under --out; %(time)s keeps re-runs from clobbering each other.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.set(
        "FEEDS",
        {
            str(out_dir / "%(name)s_%(time)s.jsonl"): {
                "format": "jsonlines",
                "encoding": "utf-8",
                "overwrite": False,
            }
        },
        priority="cmdline",
    )

    process = CrawlerProcess(settings)
    process.crawl(TurkishSitemapSpider, hosts_csv=args.hosts, min_pages=args.min_pages)
    process.start()  # blocks until the crawl finishes
    print(f"Crawl finished. JSONL written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
