"""CLI: ingest Turkish legislation (mevzuat) into raw datatrove-ready JSONL (roadmap 2).

Streams the mevzuat.gov.tr Hugging Face mirror (no scraping) into ``{"id","text","metadata"}``
records the EXISTING cleaning pipeline reads verbatim::

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/mevzuat

The SCRAPED legal sources (Resmî Gazete, TBMM transcripts — real; court decisions, YÖKTEZ —
scaffolds) live in scripts/download_govlegal.py, since they need a polite live scraper.

Examples
--------
    uv run --extra sources python scripts/ingest_govlegal.py --out output/raw/mevzuat --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources import govlegal  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_govlegal",
        description="Ingest Turkish legislation (mevzuat HF mirror) into raw JSONL. For "
        "Resmî Gazete / TBMM / courts use scripts/download_govlegal.py.",
    )
    p.add_argument(
        "--out",
        default="output/raw/govlegal",
        help="Output directory for raw JSONL (default: output/raw/govlegal).",
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
    # Per-source subdirectory keeps each source's shards separate for per-source token
    # accounting in the blend manifest.
    out_dir = str(Path(args.out) / govlegal.MEVZUAT.name)
    written = govlegal.ingest_mevzuat(out_dir, limit=args.limit)
    print(f"[mevzuat] wrote {written} record(s) to {out_dir}")
    print("next: tc-run-hplt --source jsonl --data-path", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
