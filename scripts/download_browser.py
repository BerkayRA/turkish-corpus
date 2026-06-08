"""CLI: browser-based download for gov/legal sites that block bots or render via JS (roadmap 2).

Wraps :mod:`turkish_corpus.sources.playwright_dl`, which drives a real headless Chromium via
Playwright. Use this for sources the plain ``PoliteSession`` scrapers can't reach:

- ``resmi_gazete`` — LIVE-VERIFIED: blocks bot UAs, so it needs a real browser. Requires
  ``--start-date``/``--end-date``; writes ``{"id","text","metadata"}`` JSONL the EXISTING cleaning
  pipeline reads verbatim.
- ``courts`` (Yargıtay/Danıştay JS SPA) and ``yoktez`` (session/CAPTCHA) are documented scaffolds:
  this CLI prints their implementation guidance to stderr and exits ``2`` rather than crashing.

The ``playwright`` extra installs the package only; the browser binary is a one-time install::

    uv sync --extra playwright
    uv run playwright install chromium

Examples
--------
    uv run --extra playwright python scripts/download_browser.py --source resmi_gazete \\
        --start-date 2024-01-02 --end-date 2024-01-05
    uv run --extra playwright python scripts/download_browser.py --source courts  # guidance, exit 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources import playwright_dl  # noqa: E402

_LIVE = ("resmi_gazete",)
_SCAFFOLD = ("courts", "yoktez")
_SCAFFOLD_FUNCS = {
    "courts": playwright_dl.download_court_decisions,
    "yoktez": playwright_dl.download_yoktez,
}
_INFO = {
    "resmi_gazete": playwright_dl.RESMI_GAZETE,
    "courts": playwright_dl.COURT_DECISIONS,
    "yoktez": playwright_dl.YOKTEZ,
}

# Surfaced on every run so an operator never forgets the one-time browser-binary step.
_PLAYWRIGHT_NOTE = (
    "[note] requires a one-time browser install: `uv run playwright install chromium`"
)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_browser",
        description="Browser-based download for gov/legal sites that block bots or render via JS.",
    )
    p.add_argument(
        "--source",
        choices=(*_LIVE, *_SCAFFOLD),
        required=True,
        help="Source to download (resmi_gazete is live; courts/yoktez are scaffolds).",
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
        "--headful",
        action="store_true",
        help="Disable headless mode (visible browser) — needed for CAPTCHA-gated sources.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Cap emitted records for smoke tests (-1 = all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(_PLAYWRIGHT_NOTE, file=sys.stderr)

    info = _INFO[args.source]
    # Per-source subdirectory keeps each source's shards separate for per-source token accounting.
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

    # resmi_gazete (the only live source)
    if args.start_date is None or args.end_date is None:
        raise SystemExit("resmi_gazete requires --start-date and --end-date (YYYY-MM-DD).")

    with playwright_dl.PlaywrightFetcher(headless=not args.headful) as fetcher:
        written = playwright_dl.download_resmi_gazete(
            out_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            fetcher=fetcher,
            limit=args.limit,
        )

    print(f"[{args.source}] wrote {written} record(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
