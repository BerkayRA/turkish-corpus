"""Pluggable token counting.

The corpus is sized in *tokens*, and the right token count depends on the tokenizer.
This project's whole premise is a **morphology-aware Turkish tokenizer** — multilingual
tokenizers over-fragment Turkish (a "tokenization premium" up to ~10-15x more tokens per
word than English), so token budgets must be measured with the real tokenizer, not a
generic one.

This module defines a small :class:`TokenCounter` protocol with three implementations:

- :class:`WhitespaceTokenCounter` — dependency-free baseline (also handy to *measure*
  your tokenizer's fertility: tokens-per-word = subword_count / whitespace_count).
- :class:`HFTokenCounter` — wraps any HuggingFace ``tokenizers`` tokenizer, loaded from a
  local ``tokenizer.json`` file or a Hub repo id.
- :class:`SentencePieceTokenCounter` — wraps a SentencePiece ``.model`` file.

Integration with the user's two sibling repos
----------------------------------------------
- ``BerkayRA/turkish-tokenizer`` is a pure-Python morphological **segmenter** (no integer
  vocabulary, ~500-700 words/sec). It is wired up in :mod:`turkish_corpus.morphology` for
  *fertility analysis*, not billion-scale in-pipeline counting (too slow).
- ``BerkayRA/turkish-llm`` *trains and exports* the actual subword tokenizers this corpus
  is sized against, into its ``models/`` dir:

  * SentencePiece ``sp_unigram_<V>.model`` / ``sp_morph_<V>.model`` — load via
    :class:`SentencePieceTokenCounter` (needs the optional ``sentencepiece`` extra).
  * HuggingFace ``byte_bpe_<V>.json`` and the custom ``morpheme_bpe_<V>.json`` (both in
    HF ``tokenizers`` format) — load via :class:`HFTokenCounter`.

The morphology-aware BPE the user will train (the production tokenizer) lives in
``turkish-llm``; export it to ``tokenizer.json`` and point :func:`load_token_counter` at
the path. The datatrove pipeline counts tokens via its own ``TokensCounter`` block, which
only accepts an HF ``tokenizer.json`` / Hub id (see ``config.TokenizerConfig.name_or_path``).
SentencePiece ``.model`` files and the morphological analyzer are supported by *this*
standalone counter and the fertility tooling, not the in-pipeline ``TokensCounter``.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

from .morpheme_bpe import MorphemeBPE, MorphemeBPETokenCounter

__all__ = [
    "TokenCounter",
    "WhitespaceTokenCounter",
    "HFTokenCounter",
    "SentencePieceTokenCounter",
    "MorphemeBPE",
    "MorphemeBPETokenCounter",
    "load_token_counter",
    "is_morpheme_bpe_spec",
]

# Top-level JSON key that uniquely marks the custom morpheme-BPE format (turkish-llm). An HF
# `tokenizer.json` carries `"model"`/`"version"` and only a *nested* merges, never this key.
_MORPHEME_BPE_DISCRIMINATOR = "morph_sep"


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can count tokens in a string."""

    def count(self, text: str) -> int: ...


class WhitespaceTokenCounter:
    """Count whitespace-delimited words. Dependency-free baseline / fertility denominator."""

    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


class HFTokenCounter:
    """Count subword tokens with a HuggingFace ``tokenizers`` tokenizer.

    Parameters
    ----------
    name_or_path:
        Either a path to a ``tokenizer.json`` file or a Hub repo id (e.g.
        ``"dbmdz/bert-base-turkish-cased"``). When the morphology-aware tokenizer is
        exported to ``tokenizer.json``, pass that path.
    add_special_tokens:
        Whether to include special tokens (BOS/EOS) in the count. Default False so the
        count reflects content tokens.
    """

    def __init__(self, name_or_path: str, *, add_special_tokens: bool = False) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415  (core dep, lazy for fast import)

        self.name_or_path = name_or_path
        self.add_special_tokens = add_special_tokens
        if os.path.isfile(name_or_path):
            self._tok = Tokenizer.from_file(name_or_path)
        else:
            self._tok = Tokenizer.from_pretrained(name_or_path)

    @property
    def name(self) -> str:
        return self.name_or_path

    def count(self, text: str) -> int:
        if not text:
            return 0
        enc = self._tok.encode(text, add_special_tokens=self.add_special_tokens)
        return len(enc.ids)


class SentencePieceTokenCounter:
    """Count subword tokens with a SentencePiece ``.model`` file.

    Handles the SentencePiece tokenizers exported by the ``turkish-llm`` repo
    (``sp_unigram_<V>.model``, ``sp_morph_<V>.model``). ``sentencepiece`` is an optional
    extra and imported lazily, so the pure core stays importable without it.

    Parameters
    ----------
    model_path:
        Path to a SentencePiece ``.model`` file.
    """

    def __init__(self, model_path: str) -> None:
        import sentencepiece  # noqa: PLC0415  (optional extra, lazy for fast import)

        self.model_path = model_path
        self._sp = sentencepiece.SentencePieceProcessor(model_file=model_path)

    @property
    def name(self) -> str:
        return self.model_path

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._sp.encode(text, out_type=int))


def load_token_counter(spec: str | None = None, **kwargs) -> TokenCounter:
    """Resolve a token-counter spec to an instance, dispatching by suffix and JSON shape.

    - ``None`` or ``"whitespace"`` -> :class:`WhitespaceTokenCounter`
    - a ``.model`` path            -> :class:`SentencePieceTokenCounter`
    - a ``.json`` path             -> :class:`MorphemeBPETokenCounter` when the file is the
      custom morpheme-BPE format (top-level ``"morph_sep"`` key), else :class:`HFTokenCounter`
    - a bare Hub repo id           -> :class:`HFTokenCounter`

    The ``.json`` dispatch peeks at the top-level JSON keys rather than trusting the
    extension: both the morpheme-BPE format and HF ``tokenizer.json`` use ``.json``, and only
    the morpheme-BPE format has a top-level ``"morph_sep"``. A JSON read error falls back to
    :class:`HFTokenCounter` (HF accepts arbitrary local files / Hub ids).
    """
    if spec is None or spec == "whitespace":
        return WhitespaceTokenCounter()
    if spec.endswith(".model"):
        return SentencePieceTokenCounter(spec, **kwargs)
    if spec.endswith(".json") and is_morpheme_bpe_spec(spec):
        return MorphemeBPETokenCounter(spec, **kwargs)
    # Other `.json` files and bare Hub repo ids both load through HF `tokenizers`.
    return HFTokenCounter(spec, **kwargs)


def is_morpheme_bpe_spec(spec: str) -> bool:
    """True iff ``spec`` is the custom morpheme-BPE JSON (top-level ``"morph_sep"`` key).

    The discriminator is the *file shape*, not the extension: both the morpheme-BPE format and
    an HF ``tokenizer.json`` use ``.json``, and only the morpheme-BPE format carries a top-level
    ``"morph_sep"`` (HF's is nested under ``"model"``). This is the single source of truth used
    both by :func:`load_token_counter`'s dispatch and by the in-pipeline ``TurkishTokensCounter``
    dispatch in ``pipeline.py``, so they can never diverge.

    Reads defensively: a non-path / Hub id / missing file / malformed JSON returns ``False`` so
    callers fall back to the HF loader rather than crashing on dispatch.
    """
    try:
        with open(spec, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and _MORPHEME_BPE_DISCRIMINATOR in data
