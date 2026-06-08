# Tokenizers & token counting

This corpus is sized in **tokens**, and the token count depends entirely on the tokenizer.
Multilingual tokenizers over-fragment Turkish (a "tokenization premium" up to ~10-15x more
tokens/word than English), so the budget must be measured with the **real** tokenizer — the
morphology-aware one this project is built around — not a generic baseline.

## The two sibling repos

| Repo | Role | Produces |
|------|------|----------|
| [`BerkayRA/turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer) | Pure-Python morphological **analyzer/segmenter** (`tr_api`). No integer vocabulary, ~500-700 words/sec. | Morpheme segmentation (used here for *fertility analysis*) |
| [`BerkayRA/turkish-llm`](https://github.com/BerkayRA/turkish-llm) | Trains & exports the **subword tokenizers** the corpus is counted against. | `sp_unigram_<V>.model`, `sp_morph_<V>.model` (SentencePiece); `byte_bpe_<V>.json` (HF); `morpheme_bpe_<V>.json` |

The **morpheme-aware BPE** the corpus will ultimately be tokenized with is trained in
`turkish-llm`. `turkish-tokenizer` is the *segmenter* that feeds morpheme-aware training and
serves as the linguistic ground truth for evaluating a subword vocabulary.

## How counting plugs in

`turkish_corpus.tokenizer.load_token_counter(spec)` dispatches by suffix:

| `spec` | Counter | Needs |
|--------|---------|-------|
| `None` / `"whitespace"` | `WhitespaceTokenCounter` | nothing (baseline / fertility denominator) |
| `*.json` or a Hub repo id | `HFTokenCounter` | core `tokenizers` dep |
| `*.model` | `SentencePieceTokenCounter` | `uv sync --extra sentencepiece` |

```python
from turkish_corpus.tokenizer import load_token_counter

# turkish-llm exports → drop straight in:
load_token_counter("/models/byte_bpe_32000.json").count("evlerimizden")     # HF
load_token_counter("/models/sp_morph_32000.model").count("evlerimizden")    # SentencePiece
```

### Counting tokens *inside* the datatrove pipeline

The datatrove `TokensCounter` block loads via HF `tokenizers`, so for in-pipeline counting
`config.TokenizerConfig.name_or_path` (CLI `--tokenizer`) **must be an HF `tokenizer.json`
or Hub id** — not a SentencePiece `.model`:

```bash
# Once the morpheme-aware BPE is exported to HF tokenizer.json in turkish-llm:
uv run --extra pipeline tc-run-hplt \
    --source jsonl --data-path /data/hplt/tur_Latn \
    --tokenizer /models/turkish-llm/morpheme_bpe.json \
    --output /data/corpus/hplt_tur --tasks 64
```

If your chosen tokenizer is SentencePiece (`sp_morph`), convert it to HF format (e.g. with
`transformers`' SentencePiece→fast conversion) before passing it to the pipeline, or count
tokens out-of-band with `SentencePieceTokenCounter`.

## Recommended flow

1. Train the morpheme-aware BPE (and baselines) in `turkish-llm`.
2. **Export it to HF `tokenizer.json`** (turkish-llm's `byte_bpe` export already is; convert
   SentencePiece if needed).
3. Point `tc-run-hplt --tokenizer` at that file to get the real corpus token count.
4. Use `scripts/measure_fertility.py` to compare candidates *before* committing to one.

## Measuring fertility

`scripts/measure_fertility.py` reports tokens/word for any counter, and — given a
`turkish-tokenizer` clone — morphemes/word and **subwords/morpheme** (closer to 1.0 means
the subword vocab mirrors true Turkish morphology):

```bash
# Clone the segmenter once and point at it (or set TURKISH_TOKENIZER_PATH):
uv run python scripts/measure_fertility.py \
    --tokenizer /models/sp_morph_32000.model \
    --input sample_tr.txt \
    --turkish-tokenizer-path /path/to/turkish-tokenizer
```

Output: `tokens`, `words`, `tokens/word`, and (with the segmenter) `morphemes`,
`morphemes/word`, `subwords/morpheme`. Use it to pick the tokenizer with the lowest fertility
that still respects morpheme boundaries.

## API surface

- `turkish_corpus.tokenizer` — `TokenCounter` protocol, `WhitespaceTokenCounter`,
  `HFTokenCounter`, `SentencePieceTokenCounter`, `load_token_counter`.
- `turkish_corpus.morphology` — `ensure_tr_api_importable(repo_path=None)` (adds the
  `turkish-tokenizer` repo to `sys.path`, like turkish-llm's `_tok.py`; resolves from arg or
  `TURKISH_TOKENIZER_PATH`), `MorphologicalSegmenter` (`.segment(text)`, `.count(text)`).
- `turkish_corpus.fertility` — `compute_fertility(counter, texts)`,
  `compare_fertility(counter, texts, segmenter=...)`.
