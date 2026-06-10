"""Build a production :class:`MorphemeTokenizer` artifact from word-based morph data.

Reads a morpheme-segmented corpus (the output of ``segment_morphemes_fast.py``: words are
space-separated, morphemes within a word joined by ``▁``), applies the learned BPE merges to
get each word's PIECES, then:

- counts piece frequencies → the fixed integer vocabulary (frequency-ranked). A word's FIRST
  piece is counted in its ``▁``-prefixed (fused) form (e.g. ``"▁gel"``) and later pieces in
  their bare form, matching the tokenizer's fused word-boundary scheme;
- counts surface-word frequencies and caches each surface form's BARE pieces → the fast
  ``word→pieces`` lookup table (the deployable hot path), keyed by the LOWERCASED surface;
- TRAINS the **byte-level BPE** merges for the OOV fallback: each surface word's bytes are
  converted to GPT-2 byte-chars (``bytes_to_unicode``) and ``--byte-merges`` merges are learned
  with the same pure BPE trainer as morphemes (:func:`turkish_corpus.bpe.learn_bpe`). These
  let rare / unparsed / mixed-case / punctuated words cost a few sub-word tokens instead of a
  raw-byte explosion, while still giving total + exact coverage.

The morph file is already lowercased by the analyzer, so surface forms here are the lowercased
words; casing is handled at encode time by the tokenizer's CAP/UPPER markers. Byte-BPE is
trained on those same surfaces (lowercased), which is fine — byte-chars are case-bearing and
the fallback re-encodes the ORIGINAL surface at runtime regardless.

No ``tr_api`` is needed here — the morph file already carries the per-word analysis. The
analyzer is only needed later for words absent from the table (handled lazily at encode time).

Usage:
    python scripts/build_morpheme_tokenizer.py /tmp/morph20k_fast.txt \
        --merges models_fresh/morpheme_bpe_prod.json -o models_fresh/morpheme_tokenizer \
        --vocab-size 64000 --byte-merges 16000 --table-top-n 1000000
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

# Allow running as a plain script (scripts/ is not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turkish_corpus.bpe import learn_bpe  # noqa: E402
from turkish_corpus.morpheme_bpe import MorphemeBPE  # noqa: E402
from turkish_corpus.morpheme_tokenizer import (  # noqa: E402
    MORPH_SEP,
    MorphemeTokenizer,
    bytes_to_unicode,
)


def _train_byte_merges(
    surface_freqs: collections.Counter[str], num_merges: int
) -> list[tuple[str, str]]:
    """Train byte-level BPE merges over GPT-2 byte-chars of the surface-word frequencies."""
    byte_map = bytes_to_unicode()
    word_symbols: dict[tuple[str, ...], int] = {}
    for surface, freq in surface_freqs.items():
        symbols = tuple(byte_map[b] for b in surface.encode("utf-8"))
        if symbols:
            word_symbols[symbols] = word_symbols.get(symbols, 0) + freq
    return learn_bpe(word_symbols, num_merges)


def build(
    morph_file: str,
    merges_path: str,
    *,
    vocab_size: int = 64000,
    byte_merges: int = 16000,
    table_top_n: int = 1_000_000,
) -> tuple[MorphemeTokenizer, dict]:
    """Assemble a MorphemeTokenizer + its fast table from a morph-segmented corpus.

    Returns the tokenizer (with the table set) and a stats dict. ``byte_merges`` byte-level BPE
    merges are trained from the surface-word frequencies and reserved a vocab section; the
    remaining ``vocab_size - (263 + n_byte_merges)`` ids hold the morpheme pieces.
    """
    bpe = MorphemeBPE.from_file(merges_path)
    piece_freqs: collections.Counter[str] = collections.Counter()
    surface_freqs: collections.Counter[str] = collections.Counter()
    surface_pieces: dict[str, list[str]] = {}

    t0 = time.time()
    n_tokens = 0
    with open(morph_file, encoding="utf-8") as f:
        for line in f:
            for tok in line.split():
                morphs = tok.split(MORPH_SEP)
                pieces = bpe.encode(morphs)
                if not pieces:
                    continue
                # Fused scheme: count the first piece in its ▁-prefixed (word-initial) form,
                # later pieces bare. This is what the tokenizer's encode() looks up.
                piece_freqs[MORPH_SEP + pieces[0]] += 1
                for p in pieces[1:]:
                    piece_freqs[p] += 1
                surface = "".join(morphs)
                surface_freqs[surface] += 1
                if surface not in surface_pieces:
                    surface_pieces[surface] = pieces  # BARE pieces; fusion at encode time.
                n_tokens += 1

    learned_byte_merges = _train_byte_merges(surface_freqs, byte_merges)

    tokenizer = MorphemeTokenizer.build(
        merges_path=merges_path,
        piece_freqs=piece_freqs,
        byte_merges=learned_byte_merges,
        vocab_size=vocab_size,
    )
    table = {w: surface_pieces[w] for w, _ in surface_freqs.most_common(table_top_n)}
    tokenizer.set_table(table)

    # Token coverage the table buys (by frequency), for reporting.
    covered = sum(c for w, c in surface_freqs.items() if w in table)
    stats = {
        "word_tokens": n_tokens,
        "unique_surfaces": len(surface_freqs),
        "distinct_pieces": len(piece_freqs),
        "vocab_size": vocab_size,
        "byte_bpe_merges": len(learned_byte_merges),
        "table_entries": len(table),
        "table_token_coverage": covered / n_tokens if n_tokens else 0.0,
        "build_seconds": time.time() - t0,
    }
    return tokenizer, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("morph_file", help="morpheme-segmented corpus (segment_morphemes_fast.py)")
    ap.add_argument("--merges", required=True, help="morpheme_bpe_<V>.json merges file")
    ap.add_argument("-o", "--output", required=True, help="output directory for the tokenizer")
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument(
        "--byte-merges",
        type=int,
        default=16000,
        help="byte-level BPE merges to train for the OOV fallback",
    )
    ap.add_argument("--table-top-n", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    tokenizer, stats = build(
        args.morph_file,
        args.merges,
        vocab_size=args.vocab_size,
        byte_merges=args.byte_merges,
        table_top_n=args.table_top_n,
    )
    tokenizer.save(args.output)
    print(
        f"[build_morpheme_tokenizer] {stats['word_tokens']:,} word tokens, "
        f"{stats['unique_surfaces']:,} unique surfaces, {stats['distinct_pieces']:,} distinct "
        f"pieces -> vocab {stats['vocab_size']:,} (byte-BPE {stats['byte_bpe_merges']:,}); "
        f"table {stats['table_entries']:,} entries covering "
        f"{stats['table_token_coverage']:.2%} of tokens; "
        f"{stats['build_seconds']:.0f}s -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
