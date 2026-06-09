"""Export cleaned corpus shards to plain text with a disjoint train/eval split.

The bridge from this repo's cleaned output (datatrove JSONL: ``{id,text,metadata}``) to a
tokenizer-trainer's expected input (plain text, one document per line). Used to feed the
sibling ``turkish-llm`` repo fresh, deduplicated training data with a **held-out** eval set
so tokenizer fertility is measured on UNSEEN text (not overfit to the training corpus).

Reads one or more cleaned ``final`` dirs, deduplicates by document id across them, shuffles
deterministically (``--seed``), and splits documents into ``train.txt`` + ``eval.txt`` with
zero overlap. Each document is written on a single line (internal whitespace collapsed).

    uv run python scripts/export_text_corpus.py \\
        --clean-dir /tmp/freshcorpus/clean_hplt/final \\
        --clean-dir /tmp/freshcorpus/clean_wiki/final \\
        --out-dir /tmp/freshcorpus/text --eval-fraction 0.05
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.blend import iter_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_text_corpus",
        description="Export cleaned JSONL shards to train.txt + held-out eval.txt (one doc/line).",
    )
    p.add_argument("--clean-dir", action="append", required=True, dest="clean_dirs",
                   help="A cleaned `final` dir of JSONL(.gz). Repeatable for multiple sources.")
    p.add_argument("--out-dir", required=True, help="Output dir for train.txt / eval.txt.")
    p.add_argument("--eval-fraction", type=float, default=0.05,
                   help="Fraction of documents held out for eval (default 0.05).")
    p.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    p.add_argument("--min-chars", type=int, default=200,
                   help="Skip documents shorter than this (post-collapse).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.eval_fraction < 1.0:
        raise SystemExit("--eval-fraction must be in (0, 1)")

    # Collect unique documents (dedup by id across sources), as single-line text.
    seen: set[str] = set()
    docs: list[str] = []
    for clean_dir in args.clean_dirs:
        for rec in iter_jsonl(clean_dir):
            doc_id = rec.get("id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            line = " ".join((rec.get("text") or "").split())
            if len(line) >= args.min_chars:
                docs.append(line)

    if not docs:
        raise SystemExit("no documents found in the given --clean-dir(s)")

    # Deterministic shuffle, then a clean document-level split (no overlap).
    random.Random(args.seed).shuffle(docs)
    n_eval = max(1, round(len(docs) * args.eval_fraction))
    eval_docs, train_docs = docs[:n_eval], docs[n_eval:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    (out / "eval.txt").write_text("\n".join(eval_docs) + "\n", encoding="utf-8")

    train_mb = (out / "train.txt").stat().st_size / 1e6
    eval_mb = (out / "eval.txt").stat().st_size / 1e6
    print(f"unique docs: {len(docs):,}  ->  train {len(train_docs):,} ({train_mb:.1f} MB) | "
          f"eval {len(eval_docs):,} ({eval_mb:.1f} MB)")
    print(f"written: {out}/train.txt , {out}/eval.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
