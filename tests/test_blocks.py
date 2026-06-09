"""Tests for the custom datatrove blocks, focused on ``TurkishTokensCounter``.

These require the optional ``pipeline`` extra (datatrove) and are skipped otherwise, so the
core suite still runs on a clean dev machine. They exercise the in-pipeline token counting
end-to-end with the dependency-free ``whitespace`` counter (no tr_api/HF needed), plus the
sampling logic and the public ``is_morpheme_bpe_spec`` discriminator.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("datatrove", reason="needs `uv sync --extra pipeline`")

pytestmark = pytest.mark.pipeline

from datatrove.data import Document  # noqa: E402

from turkish_corpus.blocks import TurkishTokensCounter, _stable_unit  # noqa: E402
from turkish_corpus.tokenizer import is_morpheme_bpe_spec  # noqa: E402


def _docs(*texts: str) -> list[Document]:
    return [Document(text=t, id=f"doc-{i}") for i, t in enumerate(texts)]


class TestTurkishTokensCounterWhitespace:
    def test_counts_whitespace_words_and_yields_all_docs(self):
        block = TurkishTokensCounter("whitespace")
        docs = _docs("bir iki üç", "tek", "")
        out = list(block.run(iter(docs)))

        # All docs flow through unchanged in order.
        assert [d.id for d in out] == ["doc-0", "doc-1", "doc-2"]
        # Per-doc token_count recorded for every doc at sample_rate=1.0.
        assert [d.metadata["token_count"] for d in out] == [3, 1, 0]

    def test_sample_rate_one_counts_all(self):
        block = TurkishTokensCounter("whitespace", sample_rate=1.0)
        out = list(block.run(iter(_docs("a b", "c d e"))))
        assert all("token_count" in d.metadata for d in out)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_sample_rate_rejected(self, bad):
        with pytest.raises(ValueError, match="sample_rate"):
            TurkishTokensCounter("whitespace", sample_rate=bad)

    def test_fractional_rate_is_deterministic_subset(self):
        # Many ids so a fractional rate selects a non-trivial, stable subset.
        texts = [f"word {i} here" for i in range(40)]

        def counted_ids() -> set[str]:
            docs = [Document(text=t, id=f"id-{i}") for i, t in enumerate(texts)]
            out = list(TurkishTokensCounter("whitespace", sample_rate=0.5).run(iter(docs)))
            return {d.id for d in out if "token_count" in d.metadata}

        first = counted_ids()
        second = counted_ids()
        # Stable across runs (md5-based, not builtin hash()).
        assert first == second
        # Sampling actually drops some docs (not all, not none) for 40 ids at rate 0.5.
        assert 0 < len(first) < 40


class TestStableUnit:
    def test_in_unit_interval_and_deterministic(self):
        v1 = _stable_unit("doc-0")
        v2 = _stable_unit("doc-0")
        assert v1 == v2
        assert 0.0 <= v1 < 1.0

    def test_distinct_inputs_differ(self):
        assert _stable_unit("a") != _stable_unit("b")


class TestIsMorphemeBpeSpec:
    def test_true_for_morpheme_bpe_json(self, tmp_path):
        path = tmp_path / "morpheme_bpe.json"
        path.write_text(json.dumps({"morph_sep": "▁", "merges": []}), encoding="utf-8")
        assert is_morpheme_bpe_spec(str(path)) is True

    def test_false_for_hf_shaped_json(self, tmp_path):
        path = tmp_path / "tokenizer.json"
        path.write_text(
            json.dumps({"version": "1.0", "model": {"type": "BPE", "merges": ["a b"]}}),
            encoding="utf-8",
        )
        assert is_morpheme_bpe_spec(str(path)) is False

    def test_false_for_model_path(self):
        assert is_morpheme_bpe_spec("/models/sp_morph_32000.model") is False

    def test_false_for_whitespace(self):
        assert is_morpheme_bpe_spec("whitespace") is False

    def test_false_for_hub_id(self):
        assert is_morpheme_bpe_spec("dbmdz/bert-base-turkish-cased") is False
