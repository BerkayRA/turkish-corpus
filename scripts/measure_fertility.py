"""CLI: measure a tokenizer's fertility (tokens per word) on Turkish text.

Loads any token counter supported by :func:`turkish_corpus.tokenizer.load_token_counter`
(HF ``tokenizer.json`` / Hub id, or a SentencePiece ``.model``) and reports tokens, words,
and tokens-per-word. With ``--turkish-tokenizer-path`` it also loads the ``turkish-tokenizer``
morphological segmenter to report morphemes-per-word and subwords-per-morpheme — the signal
for how close a subword vocabulary is to true Turkish morphology.

Examples
--------
HF byte-level BPE exported from turkish-llm, over a file of one doc per line::

    uv run python scripts/measure_fertility.py \\
        --tokenizer /models/byte_bpe_32000.json --input sample.txt

SentencePiece model plus morphological comparison on an inline string::

    uv run python scripts/measure_fertility.py \\
        --tokenizer /models/sp_morph_32000.model \\
        --text "evlerimizden gelenler kitaplarımı okudular" \\
        --turkish-tokenizer-path /path/to/turkish-tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.fertility import compare_fertility  # noqa: E402
from turkish_corpus.tokenizer import load_token_counter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_fertility",
        description="Measure tokenizer fertility (tokens per word) on Turkish text.",
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.json / SentencePiece .model, or a Hub repo id.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Text file, one document per line.")
    source.add_argument("--text", help="A single inline string to measure.")
    p.add_argument(
        "--turkish-tokenizer-path",
        default=None,
        help=(
            "Path to a BerkayRA/turkish-tokenizer clone to also report morphemes/word "
            "and subwords/morpheme (else set TURKISH_TOKENIZER_PATH)."
        ),
    )
    return p


def _read_texts(args: argparse.Namespace) -> list[str]:
    """Return the documents to measure from --input (one per line) or --text."""
    if args.text is not None:
        return [args.text]
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"--input file not found: {args.input}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    texts = _read_texts(args)

    counter = load_token_counter(args.tokenizer)

    segmenter = None
    if args.turkish_tokenizer_path is not None:
        # Lazy import: the segmenter pulls in tr_api, an optional sibling repo.
        from turkish_corpus.morphology import MorphologicalSegmenter

        segmenter = MorphologicalSegmenter(args.turkish_tokenizer_path)

    report = compare_fertility(counter, texts, segmenter=segmenter)

    print(f"tokenizer:        {getattr(counter, 'name', args.tokenizer)}")
    print(f"documents:        {len(texts)}")
    print(f"tokens:           {report['tokens']}")
    print(f"words:            {report['words']}")
    print(f"tokens/word:      {report['tokens_per_word']:.4f}")
    if segmenter is not None:
        words = report["words"]
        morphemes = report["morphemes"]
        print(f"morphemes:        {morphemes}")
        print(f"morphemes/word:   {(morphemes / words if words else 0.0):.4f}")
        print(f"subwords/morpheme:{report['subwords_per_morpheme']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
