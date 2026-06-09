"""The morpheme-aware BPE token counter trained in ``BerkayRA/turkish-llm``.

This is the project's *best* tokenizer: a Byte-Pair-Encoding model whose base units are
**morphemes**, not characters/bytes. Token boundaries are therefore always morpheme
boundaries (boundary precision 1.0 by construction), and learned merges of frequent adjacent
morphemes (e.g. ``-iyor-um``, ``-lar-ı-nda``) collapse into single tokens, pulling fertility
below the rigid segment-then-subword floor. See ``turkish-llm/morpheme_bpe.py``.

File format (``morpheme_bpe_<V>.json``)
---------------------------------------
A custom (non-HuggingFace) JSON with two top-level keys::

    {"morph_sep": "▁", "merges": [[a, b], ...]}

``merges`` is an *ordered* list of 2-element ``[a, b]`` string pairs; the list index is the
merge rank (lower = higher priority). The top-level ``"morph_sep"`` key is what distinguishes
this format from an HF ``tokenizer.json`` (which carries ``"model"``/``"version"`` and only a
*nested* ``merges``, never a top-level ``morph_sep``) — :func:`turkish_corpus.tokenizer.
load_token_counter` uses exactly that key to dispatch here.

Segmentation dependency
-----------------------
Counting a word means analysing it into morphemes first, which needs the
``BerkayRA/turkish-tokenizer`` repo (``tr_api``) on ``sys.path`` — located via the
``repo_path`` argument or the ``TURKISH_TOKENIZER_PATH`` env var (see
:func:`turkish_corpus.morphology.ensure_tr_api_importable`).

The analyzer is pure-Python and slow (~500-700 words/sec) *per unique form*, but Turkish text
has a ~5.5% type/token ratio (≈92% of tokens repeat a form already seen), so the per-word
``lru_cache`` (see ``cache_size``) makes the amortized cost an order of magnitude lower over a
corpus and warms up serving. For full deployment, precomputing a top-~1M ``word→pieces`` table
covers ~96% of tokens at lookup speed with the analyzer handling only the long tail. The
in-pipeline datatrove ``TokensCounter`` still requires an HF ``tokenizer.json`` (this custom
format is not HF), so count corpus tokens with the morpheme-aware BPE via this counter (e.g.
the :class:`turkish_corpus.blocks.TurkishTokensCounter` block) rather than that path.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

__all__ = [
    "MorphemeBPE",
    "MorphemeBPETokenCounter",
]


class MorphemeBPE:
    """Pure morpheme-BPE merge engine (no ``tr_api`` dependency, unit-testable).

    Replicates ``turkish-llm/morpheme_bpe.py``'s ``MorphemeBPE`` exactly: a greedy,
    lowest-rank-first merge loop over a word's morpheme list.

    Parameters
    ----------
    merges:
        Ordered list of ``(a, b)`` morpheme pairs; the list index is the merge rank (lower =
        applied first).
    """

    def __init__(self, merges: list[tuple[str, str]]) -> None:
        # rank = merge priority (lower = applied first), mirroring the trainer's output order.
        self.ranks: dict[tuple[str, str], int] = {
            (a, b): i for i, (a, b) in enumerate(merges)
        }

    @classmethod
    def from_file(cls, path: str) -> MorphemeBPE:
        """Build from a ``morpheme_bpe_<V>.json`` file (top-level ``merges`` list)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])

    def encode(self, morphemes: list[str]) -> list[str]:
        """Merge a word's morpheme list into morpheme-BPE tokens.

        At each step the mergeable adjacent pair with the lowest rank wins; ties are broken by
        leftmost position. Loops until no adjacent pair has a known merge.
        """
        sym = list(morphemes)
        while len(sym) > 1:
            best_rank, best_idx = None, None
            for j in range(len(sym) - 1):
                r = self.ranks.get((sym[j], sym[j + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_idx = r, j
            if best_idx is None:
                break
            sym = sym[:best_idx] + [sym[best_idx] + sym[best_idx + 1]] + sym[best_idx + 2 :]
        return sym


class MorphemeBPETokenCounter:
    """Count morpheme-BPE tokens for Turkish text (the production tokenizer's counts).

    Bridges the pure :class:`MorphemeBPE` merge engine to per-word morpheme segmentation from
    ``tr_api`` (the ``turkish-tokenizer`` repo), matching ``turkish-llm``'s
    ``MorphemeBPEAdapter``: each word is analysed with ``split_clitics=False`` into morpheme
    ``"chunk"`` strings (falling back to ``[word]`` when the parse fails), then merged.

    Satisfies the :class:`turkish_corpus.tokenizer.TokenCounter` protocol via :meth:`count`.

    Parameters
    ----------
    merges_path:
        Path to a ``morpheme_bpe_<V>.json`` file exported from ``turkish-llm``.
    repo_path:
        Optional path to the ``turkish-tokenizer`` clone; forwarded to
        :func:`turkish_corpus.morphology.ensure_tr_api_importable` (else resolved from
        ``TURKISH_TOKENIZER_PATH``).
    cache_size:
        Per-word ``lru_cache`` size for the (slow) tr_api analysis + merge. Turkish text has a
        ~5.5% type/token ratio (~92% of tokens are repeats of a form already seen), so caching
        per *unique* word form gives an order-of-magnitude speedup over a corpus and warms
        serving. ``None`` = unbounded (best for one-pass corpus tokenization, grows with the
        vocabulary); a bounded size (default 1,000,000 — covers ~96% of tokens by frequency)
        caps memory for long-running / serving use. ``0``/negative disables caching.
    """

    def __init__(
        self,
        merges_path: str,
        *,
        repo_path: str | None = None,
        cache_size: int | None = 1_000_000,
    ) -> None:
        from turkish_corpus.morphology import (  # noqa: PLC0415  (avoid import cycle / lazy)
            ensure_tr_api_importable,
        )

        self.merges_path = merges_path
        self._bpe = MorphemeBPE.from_file(merges_path)

        # tr_api lives in an optional sibling repo, so resolve + import it lazily here.
        ensure_tr_api_importable(repo_path)
        from tr_api import Tokenizer, TokenizerConfig  # noqa: PLC0415  (optional, lazy)

        # Disable the slow, accuracy-oriented passes; fertility counting only needs the parse.
        self._tokenizer = Tokenizer(
            TokenizerConfig(
                suggest_on_oov=False,
                include_alternatives=False,
                correct_tail_typos=False,
            )
        )

        # Memoize per unique word form (the analysis is the bottleneck; merges are cheap).
        # Bound the wrapper to this instance so the cache is GC'd with it (no global leak) and
        # keyed only on the word (merges are fixed per instance). maxsize=None when disabled
        # would never evict, so map "disabled" to a tiny dummy by skipping the cache instead.
        maxsize = cache_size if cache_size is None else max(int(cache_size), 0)
        self._cache_enabled = maxsize is None or maxsize > 0
        self._encode_word_cached = (
            functools.lru_cache(maxsize=maxsize)(self._encode_word_uncached)
            if self._cache_enabled
            else self._encode_word_uncached
        )

    @property
    def name(self) -> str:
        return self.merges_path

    def _encode_word_uncached(self, word: str) -> tuple[str, ...]:
        """Morpheme-BPE tokens for one word as an immutable tuple (matches
        ``MorphemeBPEAdapter.encode``).

        Returned immutable so the lru_cache can safely share it; ``split_clitics=False`` keeps
        the flat ``{"morphemes": [...]}`` shape (no clitic ``"segments"`` split).
        """
        analysis = self._tokenizer.tokenize(
            word,
            suggest=False,
            tail_repair=False,
            alternatives=False,
            split_clitics=False,
        )
        morphs = (
            [m["chunk"] for m in analysis.get("morphemes", [])]
            if analysis.get("parsed")
            else []
        )
        if not morphs:
            morphs = [word]
        return tuple(self._bpe.encode(morphs))

    def encode_word(self, word: str) -> list[str]:
        """Cached morpheme-BPE tokens for a single word (fresh list; safe to mutate)."""
        return list(self._encode_word_cached(word))

    def count(self, text: str) -> int:
        """Morpheme-BPE token count for ``text`` (``count("") == 0``).

        Reads the cached tuples directly (no per-call list copy) for speed.
        """
        return sum(len(self._encode_word_cached(word)) for word in text.split())

    def cache_info(self) -> functools.CacheInfo | None:
        """``functools.lru_cache`` stats (hits/misses/maxsize/currsize), or ``None`` if the
        cache is disabled — useful to confirm the expected ~90%+ hit rate on real text."""
        info = getattr(self._encode_word_cached, "cache_info", None)
        return info() if info is not None else None
