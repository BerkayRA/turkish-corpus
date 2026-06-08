"""CLI: ingest an academic source (DergiPark / YÖKTEZ) into raw datatrove-ready JSONL.

Extracts the text layer of already-downloaded PDFs under ``--pdf-dir`` and writes
``{"id","text","metadata"}`` records the EXISTING cleaning pipeline reads verbatim::

    uv run --extra pipeline tc-run-hplt --source jsonl --data-path <--out dir> \\
        --tokenizer /models/tr-morph/tokenizer.json --output /data/corpus/academic

Download is a separate, harder step (DergiPark OAI-PMH; the session/CAPTCHA-gated YÖK portal)
— see the scaffolds in :mod:`turkish_corpus.sources.academic`. This CLI is offline: it only
extracts PDFs you have already fetched into ``--pdf-dir``. Scanned (image-only) PDFs have no
text layer; they are reported as "needing OCR" rather than emitted (see the module's OCR note).

Examples
--------
    uv run --extra academic python scripts/ingest_academic.py \\
        --source dergipark --pdf-dir downloads/dergipark --limit 100
    uv run --extra academic python scripts/ingest_academic.py \\
        --source yoktez --pdf-dir downloads/yoktez
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.sources import academic  # noqa: E402

# Map the CLI's --source choices to the matching SourceInfo. Extraction is identical across
# sources (ingest_pdfs), so we only need the provenance to vary.
_DISPATCH = {
    "dergipark": academic.DERGIPARK,
    "yoktez": academic.YOKTEZ,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_academic",
        description="Ingest academic PDFs (DergiPark / YÖKTEZ) into raw datatrove-ready JSONL.",
    )
    p.add_argument(
        "--source",
        choices=sorted(_DISPATCH),
        required=True,
        help="Which academic source the PDFs belong to (sets license/provenance).",
    )
    p.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory of already-downloaded *.pdf files to extract.",
    )
    p.add_argument(
        "--out",
        default="output/raw/academic",
        help="Output directory for raw JSONL (default: output/raw/academic).",
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
    info = _DISPATCH[args.source]

    # Per-source subdirectory keeps each source's shards separate for per-source token
    # accounting in the blend manifest.
    out_dir = str(Path(args.out) / info.name)

    # Count scanned PDFs (no text layer) so the operator sees the OCR backlog. ingest_pdfs
    # logs this too, but printing it here makes the next action obvious without log plumbing.
    paths = sorted(Path(args.pdf_dir).glob("*.pdf"))
    records = academic._records_from_pdfs(
        paths, info, academic.extract_pdf_text, args.limit
    )
    from turkish_corpus.sources.base import write_records  # noqa: PLC0415

    written = write_records(records, out_dir)
    needs_ocr = getattr(records, "needs_ocr", 0)

    print(f"[{args.source}] wrote {written} record(s) to {out_dir}")
    print(f"[{args.source}] {needs_ocr} PDF(s) had no text layer (scanned) — need OCR")
    print(f"next: tc-run-hplt --source jsonl --data-path {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
