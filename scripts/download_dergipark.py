"""CLI: harvest DergiPark over OAI-PMH and download open-access article PDFs.

Builds a polite session, harvests Dublin Core records from DergiPark's OAI-PMH endpoint
(paging via ``resumptionToken``), resolves each article's full-text PDF URL, and downloads the
PDFs into ``--out-pdf-dir``. This is the network step only; turning the PDFs into cleaned text
is the separate, offline academic ingester (printed as the next step).

The OAI base URL and ``--set`` specs should be verified against a live ``?verb=Identify`` /
``?verb=ListSets`` request — see :mod:`turkish_corpus.sources.dergipark`. DergiPark articles
are mostly CC BY per-journal; verify each journal's license.

Examples
--------
    uv run --extra sources python scripts/download_dergipark.py \\
        --out-pdf-dir downloads/dergipark --max-records 50

    uv run --extra sources python scripts/download_dergipark.py \\
        --out-pdf-dir downloads/dergipark --set <journal-set-spec>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources._http import DEFAULT_USER_AGENT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_dergipark",
        description="Harvest DergiPark over OAI-PMH and download article PDFs.",
    )
    p.add_argument(
        "--out-pdf-dir",
        required=True,
        help="Directory to write downloaded *.pdf files into.",
    )
    p.add_argument(
        "--set",
        dest="set_spec",
        default=None,
        help="Optional OAI set spec to restrict the harvest (per-journal/publisher; "
        "verify available sets live with ?verb=ListSets).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=-1,
        help="Cap harvested+downloaded records for smoke tests (-1 = all).",
    )
    p.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Contactable User-Agent for the polite session.",
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
        help="Override the safety check that blocks the placeholder User-Agent contact "
        "(NOT for real harvests).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Refuse to hit live hosts while still advertising the placeholder contact — operators
    # must be able to reach a real human (same posture as the crawler / other downloaders).
    if not args.allow_placeholder_ua and "example.org" in args.user_agent:
        print(
            "User-Agent still uses the placeholder contact (example.org). Pass a real, "
            "monitored URL/email via --user-agent before a live harvest (or pass "
            "--allow-placeholder-ua to override for local testing).",
            file=sys.stderr,
        )
        return 2

    # Imported here so --help works without the heavy `sources` extra installed.
    from turkish_corpus.sources import dergipark  # noqa: PLC0415

    count = dergipark.download_dergipark(
        args.out_pdf_dir,
        set_spec=args.set_spec,
        max_records=args.max_records,
        user_agent=args.user_agent,
        min_delay=args.min_delay,
    )

    print(f"[dergipark] downloaded {count} PDF(s) to {args.out_pdf_dir}")
    print(
        "next: uv run --extra academic python scripts/ingest_academic.py "
        f"--source dergipark --pdf-dir {args.out_pdf_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
