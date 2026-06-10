"""A small, pure Byte-Pair-Encoding *trainer* (``learn_bpe``), symbol-agnostic.

This factors out the merge-learning loop so it can train merges over ANY symbol alphabet
without dragging in the segmentation / serving concerns of :mod:`turkish_corpus.morpheme_bpe`
(which only *applies* merges). It is used by the build script to learn the **byte-level BPE**
merges for the tokenizer's OOV fallback (over the 256 GPT-2 byte-chars), and is equally usable
to learn morpheme-level merges over morpheme symbols.

The output is the same ``[(a, b), ...]`` ordered merge list that
:class:`turkish_corpus.morpheme_bpe.MorphemeBPE` consumes (list index = merge rank). Counting
is over a ``word -> frequency`` map (each "word" already pre-split into its base symbols), so
the trainer never touches a corpus, ``tr_api``, or any I/O — it stays pure and unit-testable.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping

__all__ = ["learn_bpe"]

# A merge whose combined corpus frequency is below this is not worth a vocab slot (the default
# guards against learning idiosyncratic noise pairs that occur once or twice).
DEFAULT_MIN_FREQ = 2


def learn_bpe(
    word_symbols: Mapping[tuple[str, ...], int] | Iterable[tuple[tuple[str, ...], int]],
    num_merges: int,
    *,
    min_freq: int = DEFAULT_MIN_FREQ,
) -> list[tuple[str, str]]:
    """Learn up to ``num_merges`` BPE merges from pre-symbolized word frequencies.

    Parameters
    ----------
    word_symbols:
        Maps each word — already split into its base symbol tuple (e.g. byte-chars for the
        byte-BPE, or morphemes for a morpheme-BPE) — to its corpus frequency. May also be an
        iterable of ``(symbols, freq)`` pairs.
    num_merges:
        Maximum number of merges to learn. Fewer are returned if no pair reaches ``min_freq``.
    min_freq:
        Minimum combined frequency for a pair to be merged. ``<= 0`` disables the floor.

    Returns
    -------
    list[tuple[str, str]]
        Ordered merges (index = rank, lower applied first), the format
        :class:`turkish_corpus.morpheme_bpe.MorphemeBPE` consumes. Deterministic for a given
        input: among equally-frequent pairs the lexicographically smallest ``(a, b)`` wins, so
        the result never depends on dict / iteration order.
    """
    if num_merges <= 0:
        return []

    # Mutable working copy: list of symbols per word, paired with that word's frequency.
    words: list[tuple[list[str], int]] = [
        (list(symbols), freq)
        for symbols, freq in (
            word_symbols.items() if isinstance(word_symbols, Mapping) else word_symbols
        )
        if freq > 0 and len(symbols) >= 1
    ]

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pair_freqs = _count_pairs(words)
        if not pair_freqs:
            break
        # Most frequent pair; ties broken lexicographically for reproducibility.
        best_pair, best_freq = min(
            pair_freqs.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if min_freq > 0 and best_freq < min_freq:
            break
        merges.append(best_pair)
        words = [(_apply_merge(symbols, best_pair), freq) for symbols, freq in words]

    return merges


def _count_pairs(words: Iterable[tuple[list[str], int]]) -> dict[tuple[str, str], int]:
    """Frequency-weighted count of every adjacent symbol pair across all words."""
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for symbols, freq in words:
        for j in range(len(symbols) - 1):
            counts[(symbols[j], symbols[j + 1])] += freq
    return counts


def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Return a new symbol list with every non-overlapping occurrence of ``pair`` merged."""
    a, b = pair
    merged: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged
