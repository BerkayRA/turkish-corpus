"""CLI: calibrate the Turkish corpus-quality filters from Turkish Wikipedia.

The FineWeb-2 "canary language" method: instead of hand-setting the Turkish stop-word list
and the Gopher/quality thresholds, DERIVE them from a clean Turkish reference corpus. This
runner streams ``wikimedia/wikipedia`` (pre-parsed clean prose), feeds article texts into
:func:`turkish_corpus.calibration.calibrate_from_corpus`, and writes the artifacts into the
package data dir (``turkish_corpus/data/``) where :mod:`turkish_corpus.filters` loads them.

``filters.py`` falls back to its curated defaults when the artifacts are absent, so this only
ever *improves* the filters — never breaks them if not run.

Examples
--------
    uv run --extra sources python scripts/calibrate_filters.py --limit 5000
    uv run --extra sources python scripts/calibrate_filters.py \\
        --limit 20000 --dump 20231101.tr --strategy quantile --q 0.005
"""

from __future__ import annotations

import argparse
import importlib.resources
import sys
from collections.abc import Iterator
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.calibration import (  # noqa: E402
    calibrate_from_corpus,
    write_artifacts,
)


def _default_out_dir() -> str:
    """Resolve the package data dir (``turkish_corpus/data/``) as the default artifact home."""
    return str(importlib.resources.files("turkish_corpus") / "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_filters",
        description="Derive Turkish stopwords + Gopher thresholds from Turkish Wikipedia.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of articles to sample (<=0 = unlimited / full dump).",
    )
    p.add_argument(
        "--dump",
        default="20231101.tr",
        help="wikimedia/wikipedia config id, date + lang (default: 20231101.tr).",
    )
    p.add_argument(
        "--top-stopwords",
        type=int,
        default=175,
        help="Number of most-frequent words to keep as stopwords (default: 175).",
    )
    p.add_argument(
        "--strategy",
        default="quantile",
        choices=("quantile", "10tail", "meanstd"),
        help="Threshold-band derivation strategy (default: quantile).",
    )
    p.add_argument(
        "--q",
        type=float,
        default=0.005,
        help="Tail fraction for the quantile strategy (default: 0.005).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write artifacts (default: the turkish_corpus/data package dir).",
    )
    return p


def _wikipedia_texts(dump: str, limit: int) -> Iterator[str]:
    """Stream up to ``limit`` Turkish Wikipedia article bodies as plain strings.

    ``limit <= 0`` streams everything (the full dump, no cap).

    ``datasets`` is imported lazily (the ``sources`` extra) so this script imports without it.
    """
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset("wikimedia/wikipedia", dump, split="train", streaming=True)
    for i, article in enumerate(dataset):
        if 0 < limit <= i:
            return
        text = article.get("text") or ""
        if text.strip():
            yield text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or _default_out_dir()

    print(f"Calibrating from wikimedia/wikipedia {args.dump} (limit={args.limit})...")
    texts = _wikipedia_texts(args.dump, args.limit)
    stopwords, thresholds, report = calibrate_from_corpus(
        texts,
        top_n=args.top_stopwords,
        strategy=args.strategy,
        q=args.q,
    )

    write_artifacts(stopwords, thresholds, out_dir, report=report)

    print(f"Wrote artifacts to: {out_dir}")
    print(f"  stopwords:  {len(stopwords)} words")
    print(f"  documents:  {report['n_docs']}")
    print(f"  vocabulary: {report['n_unique_words']} unique words")
    print("  thresholds:")
    for key, value in report["thresholds"].items():
        print(f"    {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
