"""Optional bridge to the ``BerkayRA/turkish-tokenizer`` morphological segmenter.

``turkish-tokenizer`` is a pure-Python morphological **analyzer** (segmenter). It is *not*
pip-installable: it's made importable by adding its repo directory to ``sys.path`` (the
``turkish-llm`` repo does the same in its ``_tok.py``). Its public API is::

    from tr_api import tokenize                  # tokenize(word) -> dict
    from tr_api import Tokenizer, TokenizerConfig

where ``tokenize(word)["morphemes"]`` is a list of dicts each carrying a ``"chunk"`` (the
morpheme surface string).

Why this lives behind an optional bridge
----------------------------------------
The analyzer has **no integer vocabulary** and runs at ~500-700 words/sec — fine for
*fertility analysis* over a sample, but far too slow for billion-scale, in-pipeline token
counting. The production tokenizer is the trained morpheme-aware BPE exported from
``turkish-llm`` (use :class:`turkish_corpus.tokenizer.HFTokenCounter` /
:class:`~turkish_corpus.tokenizer.SentencePieceTokenCounter` for that). This module exists
purely so :mod:`turkish_corpus.fertility` can compute morphemes-per-word and
subwords-per-morpheme to *evaluate* a subword tokenizer against the linguistic truth.

``tr_api`` is imported lazily (only inside the methods that need it) so the pure core stays
importable without the segmenter on ``sys.path``.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "ensure_tr_api_importable",
    "MorphologicalSegmenter",
]

# Environment variable used to locate a local clone of BerkayRA/turkish-tokenizer.
TURKISH_TOKENIZER_PATH_ENV = "TURKISH_TOKENIZER_PATH"


def ensure_tr_api_importable(repo_path: str | None = None) -> None:
    """Add the ``turkish-tokenizer`` repo dir to ``sys.path`` so ``import tr_api`` works.

    Mirrors the ``sys.path.insert`` trick the ``turkish-llm`` repo uses in its ``_tok.py``.
    The repo directory is resolved in priority order:

    1. the ``repo_path`` argument, if given;
    2. the ``TURKISH_TOKENIZER_PATH`` environment variable.

    Parameters
    ----------
    repo_path:
        Path to a local clone of ``BerkayRA/turkish-tokenizer`` (the directory containing
        ``tr_api.py``).

    Raises
    ------
    FileNotFoundError
        If no path is provided and the env var is unset, or the resolved path does not
        contain ``tr_api.py``. The message explains how to clone the repo and set the var.
    """
    resolved = repo_path or os.environ.get(TURKISH_TOKENIZER_PATH_ENV)
    if not resolved:
        raise FileNotFoundError(
            "The Turkish morphological segmenter was not found. Clone "
            "https://github.com/BerkayRA/turkish-tokenizer and either pass its path as "
            "`repo_path=` or set the "
            f"{TURKISH_TOKENIZER_PATH_ENV} environment variable to the repo directory "
            "(the one containing tr_api.py)."
        )
    if not os.path.isfile(os.path.join(resolved, "tr_api.py")):
        raise FileNotFoundError(
            f"{resolved!r} does not contain tr_api.py — point "
            f"{TURKISH_TOKENIZER_PATH_ENV} (or `repo_path`) at a clone of "
            "BerkayRA/turkish-tokenizer."
        )
    if resolved not in sys.path:
        # Insert at the front so a local checkout wins over any stale entry.
        sys.path.insert(0, resolved)


class MorphologicalSegmenter:
    """Wrap ``tr_api.Tokenizer`` to segment Turkish text into morpheme chunks.

    Constructed with the *fast* config (``suggest_on_oov=False``,
    ``include_alternatives=False``, ``correct_tail_typos=False``) — the OOV-suggestion,
    alternatives, and typo-repair passes are slow and irrelevant for fertility counting.

    Satisfies the :class:`turkish_corpus.tokenizer.TokenCounter` protocol via :meth:`count`
    (morpheme count), so it can be dropped into the same fertility helpers as the subword
    counters. Remember it measures *morphemes*, not the production tokenizer's subwords;
    it's a denominator for "subwords per morpheme", not a billion-scale counter.

    Parameters
    ----------
    repo_path:
        Optional path to the ``turkish-tokenizer`` clone; forwarded to
        :func:`ensure_tr_api_importable` (else resolved from ``TURKISH_TOKENIZER_PATH``).
    """

    name = "morphological-segmenter"

    def __init__(self, repo_path: str | None = None) -> None:
        ensure_tr_api_importable(repo_path)
        from tr_api import Tokenizer, TokenizerConfig  # noqa: PLC0415  (optional, lazy)

        # Disable the slow, accuracy-oriented passes; we only need fast segmentation.
        config = TokenizerConfig(
            suggest_on_oov=False,
            include_alternatives=False,
            correct_tail_typos=False,
        )
        self._tokenizer = Tokenizer(config)

    def segment(self, text: str) -> list[str]:
        """Return the morpheme chunks for every word in ``text``.

        Punctuation and whitespace carry no morphemes and are skipped; the result is a flat
        list of morpheme surfaces (the ``"chunk"`` field) across all words, in order.
        """
        if not text:
            return []
        chunks: list[str] = []
        # `tokenize_batch` over whitespace-split words gives a stable per-word dict whose
        # shape we control, avoiding the richer (and noisier) `tokenize_text` token stream.
        for analysis in self._tokenizer.tokenize_batch(text.split()):
            chunks.extend(_chunks_from_analysis(analysis))
        return chunks

    def count(self, text: str) -> int:
        """Morpheme count for ``text`` (``count("") == 0``)."""
        return len(self.segment(text))


def _chunks_from_analysis(analysis: dict) -> list[str]:
    """Pull morpheme ``"chunk"`` strings out of one ``tr_api`` word analysis.

    Handles both shapes ``tr_api`` can return: the flat ``{"morphemes": [...]}`` and the
    clitic-split ``{"segments": [{...}, ...]}`` (e.g. ``gelecekmisin`` -> gelecek + misin),
    where each segment carries its own ``"morphemes"``.
    """
    if "segments" in analysis:
        chunks: list[str] = []
        for segment in analysis["segments"]:
            chunks.extend(_chunks_from_analysis(segment))
        return chunks
    morphemes = analysis.get("morphemes") or []
    return [m["chunk"] for m in morphemes if "chunk" in m]
