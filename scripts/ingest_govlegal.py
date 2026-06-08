"""CLI: ingest a Turkish government/legal source into raw datatrove-ready JSONL (roadmap 2).

Dispatches ``--source`` to the matching ingester in
:mod:`turkish_corpus.sources.govlegal` and writes ``{"id","text","metadata"}`` records that
the EXISTING cleaning pipeline reads verbatim::

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/mevzuat

Only ``mevzuat`` is live (streams a Hugging Face mirror — no scraping). The scraped sources
(``resmi_gazete``, ``courts``, ``tbmm``) are scaffolds: this CLI prints their implementation
guidance and exits ``2`` rather than crashing with a traceback.

Examples
--------
    uv run --extra sources python scripts/ingest_govlegal.py --source mevzuat --limit 100
    uv run --extra sources python scripts/ingest_govlegal.py --source courts  # guidance, exit 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources import govlegal  # noqa: E402

# Map the CLI's --source choices to (ingester, source-info). The CLI name "courts" reads
# better than the module's "court_decisions"; the indirection keeps the two decoupled.
_DISPATCH = {
    "mevzuat": (govlegal.ingest_mevzuat, govlegal.MEVZUAT),
    "resmi_gazete": (govlegal.ingest_resmi_gazete, govlegal.RESMI_GAZETE),
    "courts": (govlegal.ingest_court_decisions, govlegal.COURT_DECISIONS),
    "tbmm": (govlegal.ingest_tbmm_tutanak, govlegal.TBMM_TUTANAK),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_govlegal",
        description="Ingest a Turkish government/legal source into raw datatrove-ready JSONL.",
    )
    p.add_argument(
        "--source",
        choices=sorted(_DISPATCH),
        default="mevzuat",
        help="Which legal source to ingest (default: mevzuat; the others are scaffolds).",
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
    ingest, info = _DISPATCH[args.source]

    # Per-source subdirectory keeps each source's shards separate for per-source token
    # accounting in the blend manifest.
    out_dir = str(Path(args.out) / info.name)

    try:
        written = ingest(out_dir, limit=args.limit)
    except NotImplementedError as exc:
        # Scaffold sources: surface the docstring guidance, don't dump a traceback. Exit 2
        # distinguishes "not built yet" from a real runtime failure.
        print(f"[{args.source}] not implemented yet:\n{exc}", file=sys.stderr)
        return 2

    print(f"[{args.source}] wrote {written} record(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
