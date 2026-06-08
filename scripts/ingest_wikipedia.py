"""CLI: ingest Turkish Wikipedia (Vikipedi) into raw datatrove-ready JSONL (roadmap step 2).

Streams the ``wikimedia/wikipedia`` dataset (pre-parsed clean prose, CC BY-SA share-alike)
and writes one ``{"id","text","metadata"}`` record per article into ``--out``; the EXISTING
pipeline then cleans it:

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/wikipedia_tur

The ``--dump`` value is a date-coded config id (e.g. ``20231101.tr``); verify the current
configs on https://huggingface.co/datasets/wikimedia/wikipedia before pinning one.

Examples
--------
    uv run --extra sources python scripts/ingest_wikipedia.py --limit 1000
    uv run --extra sources python scripts/ingest_wikipedia.py \\
        --out output/raw/wikipedia --dump 20231101.tr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources.wikipedia import ingest_wikipedia  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_wikipedia",
        description="Ingest Turkish Wikipedia into raw datatrove-ready JSONL.",
    )
    p.add_argument(
        "--out",
        default="output/raw/wikipedia",
        help="Output directory for raw JSONL (default: output/raw/wikipedia).",
    )
    p.add_argument(
        "--dump",
        default="20231101.tr",
        help="wikimedia/wikipedia config id, date + lang (default: 20231101.tr).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max articles to ingest; <=0 means all (default: -1).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = ingest_wikipedia(args.out, dump=args.dump, limit=args.limit)
    print(f"Wrote {written} Wikipedia records to: {args.out}")
    print(f"next: tc-run-hplt --source jsonl --data-path {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
