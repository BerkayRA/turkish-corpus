"""Tests for the pluggable token counter."""

from unittest.mock import MagicMock, patch

from turkish_corpus.tokenizer import (
    HFTokenCounter,
    TokenCounter,
    WhitespaceTokenCounter,
    load_token_counter,
)


class TestWhitespaceTokenCounter:
    def test_counts_words(self):
        assert WhitespaceTokenCounter().count("bir iki üç") == 3

    def test_empty_is_zero(self):
        assert WhitespaceTokenCounter().count("") == 0

    def test_collapses_runs(self):
        assert WhitespaceTokenCounter().count("a   b\t c\n d") == 4


class TestLoadTokenCounter:
    def test_none_returns_whitespace(self):
        assert isinstance(load_token_counter(None), WhitespaceTokenCounter)

    def test_whitespace_spec(self):
        assert isinstance(load_token_counter("whitespace"), WhitespaceTokenCounter)

    def test_satisfies_protocol(self):
        counter = load_token_counter(None)
        assert isinstance(counter, TokenCounter)

    def test_path_routes_to_hf_counter(self, tmp_path):
        tok_file = tmp_path / "tokenizer.json"
        tok_file.write_text("{}")
        mock_tok = MagicMock()
        mock_tok.encode.return_value = MagicMock(ids=[1, 2, 3])
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            counter = load_token_counter(str(tok_file))
            assert isinstance(counter, HFTokenCounter)


class TestHFTokenCounter:
    def test_counts_subwords_from_file(self, tmp_path):
        tok_file = tmp_path / "tokenizer.json"
        tok_file.write_text("{}")
        mock_tok = MagicMock()
        mock_tok.encode.return_value = MagicMock(ids=[10, 20, 30, 40])
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok) as m:
            counter = HFTokenCounter(str(tok_file))
            assert counter.count("evlerimizden gelenler") == 4
            assert counter.name == str(tok_file)
            m.assert_called_once()

    def test_empty_string_short_circuits(self, tmp_path):
        tok_file = tmp_path / "tokenizer.json"
        tok_file.write_text("{}")
        mock_tok = MagicMock()
        with patch("tokenizers.Tokenizer.from_file", return_value=mock_tok):
            counter = HFTokenCounter(str(tok_file))
            assert counter.count("") == 0
            mock_tok.encode.assert_not_called()
