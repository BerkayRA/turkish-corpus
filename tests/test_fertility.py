"""Tests for tokenizer fertility analysis."""

from turkish_corpus.fertility import compare_fertility, compute_fertility
from turkish_corpus.tokenizer import WhitespaceTokenCounter


class _FakeCounter:
    """A counter that returns a fixed token count per call, for arithmetic tests."""

    def __init__(self, per_text: int) -> None:
        self.per_text = per_text

    def count(self, text: str) -> int:
        return self.per_text if text else 0


class _FakeSegmenter:
    """Stand-in for MorphologicalSegmenter: fixed morpheme count per text."""

    def __init__(self, per_text: int) -> None:
        self.per_text = per_text

    def count(self, text: str) -> int:
        return self.per_text if text else 0


class TestComputeFertility:
    def test_whitespace_counter_is_one_token_per_word(self):
        texts = ["bir iki üç", "dört beş"]
        report = compute_fertility(WhitespaceTokenCounter(), texts)
        assert report["tokens"] == 5
        assert report["words"] == 5
        assert report["tokens_per_word"] == 1.0

    def test_tokens_per_word_math(self):
        # 2 texts × 3 tokens = 6 tokens; "evlerimizden gelenler" etc. -> 4 words total.
        texts = ["evlerimizden gelenler", "kitaplarımı okudular"]
        report = compute_fertility(_FakeCounter(3), texts)
        assert report["tokens"] == 6
        assert report["words"] == 4
        assert report["tokens_per_word"] == 1.5

    def test_empty_input_no_zero_division(self):
        report = compute_fertility(_FakeCounter(3), [])
        assert report == {"tokens": 0, "words": 0, "tokens_per_word": 0.0}


class TestCompareFertility:
    def test_without_segmenter_matches_compute(self):
        texts = ["bir iki üç"]
        assert compare_fertility(WhitespaceTokenCounter(), texts) == compute_fertility(
            WhitespaceTokenCounter(), texts
        )

    def test_with_segmenter_reports_subwords_per_morpheme(self):
        texts = ["evlerimizden", "gelenler"]
        # subword counter: 4 per text -> 8 tokens; segmenter: 2 morphemes per text -> 4.
        report = compare_fertility(
            _FakeCounter(4), texts, segmenter=_FakeSegmenter(2)
        )
        assert report["tokens"] == 8
        assert report["morphemes"] == 4
        assert report["subwords_per_morpheme"] == 2.0

    def test_segmenter_with_zero_morphemes_no_zero_division(self):
        report = compare_fertility(_FakeCounter(4), [""], segmenter=_FakeSegmenter(0))
        assert report["subwords_per_morpheme"] == 0.0
