"""Command-line entry point: run the HPLT v2 Turkish cleaning pipeline.

Examples
--------
Small local dev slice (streams 2000 docs from the Hub, no tokenizer count)::

    uv run --extra pipeline tc-run-hplt --limit 2000 --output ./output/dev --tasks 2

Full run from pre-downloaded JSONL shards, counting tokens with your tokenizer::

    uv run --extra pipeline tc-run-hplt \\
        --source jsonl --data-path /data/hplt/tur_Latn \\
        --tokenizer /models/tr-morph/tokenizer.json \\
        --output /data/corpus/hplt_tur --tasks 64
"""

from __future__ import annotations

import argparse
import logging

from .config import PipelineConfig, default_hplt_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tc-run-hplt",
        description="Clean HPLT v2 Turkish (tur_Latn) into a high-quality corpus.",
    )
    p.add_argument("--source", choices=["hf", "jsonl"], default="hf",
                   help="Read from the HuggingFace Hub or local JSONL shards.")
    p.add_argument("--data-path", default=None,
                   help="Folder of JSONL(.zst) shards (required when --source jsonl).")
    p.add_argument("--limit", type=int, default=-1,
                   help="Max documents to read (-1 = all). Use a small value for dev.")
    p.add_argument("--output", default="./output/hplt_tur", help="Output base folder.")
    p.add_argument("--logging-dir", default=None, help="Logs folder (default: <output>/logs).")
    p.add_argument("--tokenizer", default=None,
                   help="Path to tokenizer.json or Hub id for token counting (optional).")
    p.add_argument("--tasks", type=int, default=4, help="Parallel tasks.")
    p.add_argument("--workers", type=int, default=-1, help="Worker processes (-1 = tasks).")
    p.add_argument("--no-language-filter", action="store_true",
                   help="Disable the secondary fastText language gate.")
    p.add_argument("--fix-mojibake", action="store_true",
                   help="Run ftfy encoding repair (needs --extra encoding).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and validate the config, print it, and exit without running.")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = default_hplt_config()
    cfg.reader.source = args.source
    cfg.reader.data_path = args.data_path
    cfg.reader.limit = args.limit
    cfg.output_path = args.output
    cfg.logging_dir = args.logging_dir or f"{args.output}/logs"
    cfg.tokenizer.name_or_path = args.tokenizer
    cfg.tasks = args.tasks
    cfg.workers = args.workers
    cfg.language.enabled = not args.no_language_filter
    cfg.fix_mojibake = args.fix_mojibake
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.dry_run:
        print("Validated pipeline configuration:")
        print(cfg)
        return 0

    # Imported here so --dry-run and --help work without the heavy pipeline extra.
    from .pipeline import run_local

    run_local(cfg)
    print(f"Done. Final corpus written under: {cfg.output_path}/final")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
