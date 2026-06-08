"""Pluggable token counting.

The corpus is sized in *tokens*, and the right token count depends on the tokenizer.
This project's whole premise is a **morphology-aware Turkish tokenizer** — multilingual
tokenizers over-fragment Turkish (a "tokenization premium" up to ~10-15x more tokens per
word than English), so token budgets must be measured with the real tokenizer, not a
generic one.

This module defines a small :class:`TokenCounter` protocol with two implementations:

- :class:`WhitespaceTokenCounter` — dependency-free baseline (also handy to *measure*
  your tokenizer's fertility: tokens-per-word = hf_count / whitespace_count).
- :class:`HFTokenCounter` — wraps any HuggingFace ``tokenizers`` tokenizer, loaded from a
  local ``tokenizer.json`` file or a Hub repo id.

When the morphology-aware tokenizer is ready, either (a) export it to ``tokenizer.json``
and point :func:`load_token_counter` at the path, or (b) add a thin adapter class that
satisfies the protocol. The datatrove pipeline counts tokens via its own
``TokensCounter`` block (see ``config.TokenizerConfig.name_or_path``); this module is the
standalone / test-time counterpart and the integration seam.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

__all__ = [
    "TokenCounter",
    "WhitespaceTokenCounter",
    "HFTokenCounter",
    "load_token_counter",
]


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


def load_token_counter(spec: str | None = None, **kwargs) -> TokenCounter:
    """Resolve a token-counter spec to an instance.

    - ``None`` or ``"whitespace"`` -> :class:`WhitespaceTokenCounter`
    - a path or Hub repo id        -> :class:`HFTokenCounter`
    """
    if spec is None or spec == "whitespace":
        return WhitespaceTokenCounter()
    return HFTokenCounter(spec, **kwargs)
