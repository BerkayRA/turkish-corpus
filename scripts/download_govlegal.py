"""CLI: scrape a Turkish government/legal source into raw datatrove-ready JSONL (roadmap 2).

Dispatches ``--source`` to the matching scraper in
:mod:`turkish_corpus.sources.govscrape` and writes ``{"id","text","metadata"}`` records that
the EXISTING cleaning pipeline reads verbatim::

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/govlegal

Live sources: ``resmi_gazete`` (needs ``--start-date``/``--end-date``) and ``tbmm`` (needs
``--terms``). The JS/CAPTCHA-gated sources (``courts``, ``yoktez``) are documented scaffolds:
this CLI prints their implementation guidance to stderr and exits ``2`` rather than crashing.

Politeness: requests are throttled, retried and robots-aware via ``PoliteSession``. The
placeholder User-Agent contact (example.org) is refused for live scraping unless you pass
``--allow-placeholder-ua`` (local testing only), mirroring scripts/run_crawl.py.

Examples
--------
    uv run --extra sources python scripts/download_govlegal.py --source resmi_gazete \\
        --start-date 2024-01-02 --end-date 2024-01-05 --user-agent "MyBot (+https://x; me@x)"
    uv run --extra sources python scripts/download_govlegal.py --source tbmm --terms 27 28 \\
        --user-agent "MyBot (+https://x; me@x)"
    uv run --extra sources python scripts/download_govlegal.py --source courts  # guidance, exit 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources import govscrape  # noqa: E402
from turkish_corpus.sources._http import DEFAULT_USER_AGENT, PoliteSession  # noqa: E402

# CLI --source choices. "courts"/"yoktez" read better than the module's longer names; the
# indirection keeps the CLI decoupled from the function names.
_LIVE = ("resmi_gazete", "tbmm")
_SCAFFOLD = ("courts", "yoktez")
_SCAFFOLD_FUNCS = {
    "courts": govscrape.download_court_decisions,
    "yoktez": govscrape.download_yoktez,
}
_INFO = {
    "resmi_gazete": govscrape.RESMI_GAZETE,
    "tbmm": govscrape.TBMM_TUTANAK,
    "courts": govscrape.COURT_DECISIONS,
    "yoktez": govscrape.YOKTEZ,
}


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_govlegal",
        description="Scrape a Turkish government/legal source into raw datatrove-ready JSONL.",
    )
    p.add_argument(
        "--source",
        choices=(*_LIVE, *_SCAFFOLD),
        required=True,
        help="Which legal source to scrape (resmi_gazete/tbmm are live; courts/yoktez scaffolds).",
    )
    p.add_argument(
        "--out",
        default="output/raw/govlegal",
        help="Output directory for raw JSONL (default: output/raw/govlegal).",
    )
    p.add_argument(
        "--start-date",
        type=_parse_date,
        help="Resmî Gazete range start (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        type=_parse_date,
        help="Resmî Gazete range end (YYYY-MM-DD).",
    )
    p.add_argument(
        "--terms",
        nargs="+",
        help="TBMM legislative term (dönem) ids to scrape, e.g. --terms 27 28.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Cap emitted records for smoke tests (-1 = all).",
    )
    p.add_argument(
        "--user-agent",
        default=None,
        help="User-Agent for the polite session (default: the package placeholder UA).",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between requests to the same host (default: 1.0).",
    )
    p.add_argument(
        "--allow-placeholder-ua",
        action="store_true",
        help="Override the safety check blocking the placeholder UA contact (NOT for real runs).",
    )
    return p


def _build_session(args: argparse.Namespace) -> PoliteSession:
    """Build a PoliteSession, refusing the placeholder contact for live scraping.

    Refusing the placeholder (example.org) contact mirrors scripts/run_crawl.py: site operators
    must be able to reach a real human (RFC 9309 etiquette + KVKK posture). ``--user-agent``
    sets a real contact; ``--allow-placeholder-ua`` overrides for local testing only.
    """
    user_agent = args.user_agent or DEFAULT_USER_AGENT
    if not args.allow_placeholder_ua and "example.org" in user_agent:
        raise SystemExit(
            "User-Agent still uses the placeholder contact (example.org). Pass --user-agent "
            "with a real, monitored URL/email before scraping a live gov host "
            "(or --allow-placeholder-ua to override for local testing)."
        )
    return PoliteSession(user_agent=user_agent, min_delay=args.min_delay)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    info = _INFO[args.source]

    # Per-source subdirectory keeps each source's shards separate for per-source token
    # accounting in the blend manifest.
    out_dir = str(Path(args.out) / info.name)

    if args.source in _SCAFFOLD:
        # Scaffold sources: surface the docstring guidance to stderr, don't crash. Exit 2
        # distinguishes "not built yet" from a real runtime failure.
        try:
            _SCAFFOLD_FUNCS[args.source](out_dir, limit=args.limit)
        except NotImplementedError as exc:
            print(f"[{args.source}] not implemented yet:\n{exc}", file=sys.stderr)
            return 2
        return 2  # pragma: no cover — scaffolds always raise

    session = _build_session(args)

    if args.source == "resmi_gazete":
        if args.start_date is None or args.end_date is None:
            raise SystemExit("resmi_gazete requires --start-date and --end-date (YYYY-MM-DD).")
        written = govscrape.ingest_resmi_gazete(
            out_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            session=session,
            limit=args.limit,
        )
    else:  # tbmm
        if not args.terms:
            raise SystemExit("tbmm requires --terms (one or more dönem ids).")
        written = govscrape.ingest_tbmm_tutanak(
            out_dir,
            terms=args.terms,
            session=session,
            limit=args.limit,
        )

    print(f"[{args.source}] wrote {written} record(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
