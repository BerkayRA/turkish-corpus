"""Tests for the pure BPE trainer (:func:`turkish_corpus.bpe.learn_bpe`).

The trainer is symbol-agnostic; these exercise it over plain-letter "words" (standing in for
either byte-chars or morphemes) so they need no corpus or ``tr_api``. The learned merges are
fed through :class:`turkish_corpus.morpheme_bpe.MorphemeBPE` to confirm they are the exact
format that engine consumes.
"""

from __future__ import annotations

import collections

from turkish_corpus.bpe import learn_bpe
from turkish_corpus.morpheme_bpe import MorphemeBPE


class TestLearnBPE:
    def test_learns_most_frequent_pair_first(self):
        # "ab" appears 10x, "bc" 3x -> first merge must be ("a", "b").
        word_freqs = {("a", "b", "c"): 3, ("a", "b"): 7}
        merges = learn_bpe(word_freqs, num_merges=1)
        assert merges == [("a", "b")]

    def test_learns_in_descending_frequency(self):
        word_freqs = {("a", "b", "c"): 5}  # ab:5, bc:5 -> tie broken lexicographically
        merges = learn_bpe(word_freqs, num_merges=2)
        # ("a","b") < ("b","c") lexicographically -> learned first; then ("ab","c").
        assert merges == [("a", "b"), ("ab", "c")]

    def test_respects_num_merges_cap(self):
        word_freqs = {("a", "b", "c", "d"): 10}
        assert len(learn_bpe(word_freqs, num_merges=2)) == 2

    def test_min_freq_floor_stops_learning(self):
        # Each pair occurs once; with the default min_freq=2 nothing is learned.
        word_freqs = {("a", "b"): 1, ("c", "d"): 1}
        assert learn_bpe(word_freqs, num_merges=10) == []

    def test_min_freq_zero_disables_floor(self):
        word_freqs = {("a", "b"): 1}
        assert learn_bpe(word_freqs, num_merges=1, min_freq=0) == [("a", "b")]

    def test_zero_num_merges_returns_empty(self):
        assert learn_bpe({("a", "b"): 100}, num_merges=0) == []

    def test_accepts_counter_and_iterable(self):
        counter = collections.Counter({("a", "b"): 5})
        assert learn_bpe(counter, num_merges=1) == [("a", "b")]
        assert learn_bpe([(("a", "b"), 5)], num_merges=1) == [("a", "b")]

    def test_deterministic_independent_of_order(self):
        a = learn_bpe({("a", "b"): 5, ("c", "d"): 5}, num_merges=2)
        b = learn_bpe({("c", "d"): 5, ("a", "b"): 5}, num_merges=2)
        assert a == b

    def test_output_consumable_by_morpheme_bpe(self):
        word_freqs = {("f", "o", "o"): 10}
        merges = learn_bpe(word_freqs, num_merges=2)
        engine = MorphemeBPE(merges)
        # The learned merges should collapse ["f","o","o"] toward a single symbol.
        assert engine.encode(["f", "o", "o"]) == ["foo"]

    def test_ignores_zero_and_single_symbol_words(self):
        word_freqs = {("a",): 100, (): 100, ("a", "b"): 3}
        assert learn_bpe(word_freqs, num_merges=1) == [("a", "b")]
