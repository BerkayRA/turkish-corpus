"""Tokenizer *fertility* analysis — pure, dependency-free.

Fertility = average tokens emitted per (whitespace) word. It is the headline metric for
how well a tokenizer fits a language: multilingual tokenizers over-fragment Turkish
(fertility well above 2-3), inflating both training cost and context usage. A morphology-
aware tokenizer should land much closer to the language's true morpheme count.

This module computes fertility for any :class:`turkish_corpus.tokenizer.TokenCounter`, and
compares a subword counter against:

- the whitespace baseline (subwords per word), and
- the morphological segmenter (subwords per morpheme), when one is provided — the closer to
  1.0, the better the subword vocabulary mirrors real Turkish morphology.

Everything here operates on the ``TokenCounter`` protocol (a ``count(text) -> int`` method)
and plain numbers, so it needs no heavy dependencies and is trivially testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the protocol/heavy types at runtime
    from .morphology import MorphologicalSegmenter
    from .tokenizer import TokenCounter

__all__ = [
    "compute_fertility",
    "compare_fertility",
]


def compute_fertility(counter: TokenCounter, texts: Iterable[str]) -> dict:
    """Compute a counter's fertility over ``texts``.

    Parameters
    ----------
    counter:
        Anything satisfying the :class:`~turkish_corpus.tokenizer.TokenCounter` protocol.
    texts:
        An iterable of documents/lines. Consumed once.

    Returns
    -------
    dict
        ``{"tokens": int, "words": int, "tokens_per_word": float}`` where ``words`` counts
        whitespace-delimited words. ``tokens_per_word`` is ``0.0`` when there are no words
        (avoids a ZeroDivisionError on empty input).
    """
    total_tokens = 0
    total_words = 0
    for text in texts:
        total_tokens += counter.count(text)
        total_words += len(text.split())
    return {
        "tokens": total_tokens,
        "words": total_words,
        "tokens_per_word": total_tokens / total_words if total_words else 0.0,
    }


def compare_fertility(
    counter: TokenCounter,
    texts: Iterable[str],
    *,
    segmenter: MorphologicalSegmenter | None = None,
) -> dict:
    """Compare a subword counter's fertility against whitespace and (optionally) morphology.

    Parameters
    ----------
    counter:
        The subword tokenizer counter under evaluation.
    texts:
        Documents to measure. Materialized into a list so it can be scanned twice (once for
        the subword counter, once for the morphological segmenter).
    segmenter:
        Optional :class:`~turkish_corpus.morphology.MorphologicalSegmenter`. When given, the
        report adds the morpheme count and the ``subwords_per_morpheme`` ratio — the signal
        for how close the subword vocabulary is to true Turkish morphology (closer to 1.0
        is better).

    Returns
    -------
    dict
        Always: ``tokens``, ``words``, ``tokens_per_word``. With a segmenter, additionally:
        ``morphemes`` and ``subwords_per_morpheme``.
    """
    docs = list(texts)
    report = compute_fertility(counter, docs)
    if segmenter is not None:
        morphemes = sum(segmenter.count(text) for text in docs)
        report["morphemes"] = morphemes
        report["subwords_per_morpheme"] = (
            report["tokens"] / morphemes if morphemes else 0.0
        )
    return report
