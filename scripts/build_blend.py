"""CLI: blend multiple CLEANED sources into the final register-diverse corpus (roadmap last step).

Each ``--source`` points at a cleaned JSONL dir (the ``final`` output of the cleaning
pipeline) with a desired share of the token budget. Weights are normalized, so they need
not sum to 1. Token budgets are measured with the pluggable counter: omit ``--tokenizer``
to size with the dependency-free whitespace counter (works today), or pass the exported
morpheme-aware ``tokenizer.json`` once it's trained.

Examples
--------
    uv run python scripts/build_blend.py \\
        --source wikipedia=output/clean/wikipedia/final:0.12 \\
        --source govlegal=output/clean/govlegal/final:0.30 \\
        --source web=output/clean/web/final:0.58 \\
        --target-tokens 15_000_000_000 \\
        --tokenizer /models/tr-morph/tokenizer.json \\
        --out output/blend

    # Whitespace-sized dev slice (no tokenizer needed):
    uv run python scripts/build_blend.py \\
        --source wikipedia=output/clean/wikipedia/final:1 --target-tokens 5e6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.blend import BlendSource, build_blend  # noqa: E402


def _parse_source(spec: str) -> BlendSource:
    """Parse ``name=PATH:WEIGHT`` into a :class:`BlendSource`.

    ``name`` and ``WEIGHT`` are split off by the first ``=`` and last ``:`` so Windows-style
    or colon-containing paths survive. Fails fast with a clear message on malformed specs.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--source must be name=PATH:WEIGHT, got {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError(f"--source must be name=PATH:WEIGHT, got {spec!r}")
    path, weight_str = rest.rsplit(":", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--source name and PATH must be non-empty: {spec!r}")
    try:
        weight = float(weight_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--source weight must be a number: {weight_str!r}"
        ) from exc
    return BlendSource(name=name, path=path, weight=weight)


def _parse_token_budget(value: str) -> int:
    """Parse a token budget like ``15_000_000_000``, ``15e9``, or ``15000000000`` to int.

    Accepts underscores (PEP 515 style) and scientific notation; rejects non-positive values.
    Plain integers are parsed as ``int`` first to avoid float precision loss on huge values
    (``int(float("..."))`` would round past 2^53); only scientific-notation forms like
    ``15e9`` fall back to the float path.
    """
    cleaned = value.replace("_", "").strip()
    try:
        tokens = int(cleaned)
    except ValueError:
        try:
            tokens = int(float(cleaned))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--target-tokens must be a number: {value!r}"
            ) from exc
    if tokens <= 0:
        raise argparse.ArgumentTypeError(f"--target-tokens must be positive, got {value!r}")
    return tokens


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_blend",
        description="Blend cleaned sources into the final corpus at a token budget.",
    )
    p.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_source,
        metavar="name=PATH:WEIGHT",
        help="A cleaned source dir with its budget share. Repeatable. Weights are normalized.",
    )
    p.add_argument(
        "--target-tokens",
        required=True,
        type=_parse_token_budget,
        help="Total token budget. Accepts 15_000_000_000 or 15e9 forms.",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json / .model path or Hub id. Omit for the whitespace counter.",
    )
    p.add_argument(
        "--out",
        default="output/blend",
        help="Output root (blend shards in <out>/blend, manifest in <out>/manifest.json).",
    )
    p.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed (default 0).")
    return p


def _print_summary(manifest: dict) -> None:
    """Print a per-source and total achieved/target token summary."""
    totals = manifest["totals"]
    print(f"tokenizer:     {totals['tokenizer']}")
    print(f"{'source':<16}{'achieved':>16}{'target':>16}{'docs':>12}{'shortfall':>16}")
    for s in manifest["sources"]:
        print(
            f"{s['name']:<16}{s['achieved_tokens']:>16,}{s['target_tokens']:>16,}"
            f"{s['docs']:>12,}{s['shortfall_tokens']:>16,}"
        )
    print(
        f"{'TOTAL':<16}{totals['achieved_tokens']:>16,}{totals['target_tokens']:>16,}"
        f"{totals['docs']:>12,}{totals['shortfall_tokens']:>16,}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_blend(
        args.source,
        args.out,
        target_tokens=args.target_tokens,
        tokenizer_spec=args.tokenizer,
        shuffle_seed=args.seed,
    )
    _print_summary(manifest)
    print(f"\nblend shards: {Path(args.out) / 'blend'}")
    print(f"manifest:     {Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
