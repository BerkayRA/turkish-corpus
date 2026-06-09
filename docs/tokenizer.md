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
| `morpheme_bpe_<V>.json` (top-level `"morph_sep"`) | `MorphemeBPETokenCounter` | `tr_api` via `TURKISH_TOKENIZER_PATH` |
| other `*.json` or a Hub repo id | `HFTokenCounter` | core `tokenizers` dep |
| `*.model` | `SentencePieceTokenCounter` | `uv sync --extra sentencepiece` |

`.json` dispatch peeks at the file's **top-level JSON keys**, not just the extension: the
custom `morpheme_bpe_<V>.json` carries a top-level `"morph_sep"` key, whereas an HF
`tokenizer.json` carries `"model"`/`"version"` and only a *nested* `merges`. So the
morpheme-aware BPE (turkish-llm's best tokenizer) now loads via `load_token_counter` for
standalone fertility / sizing — it needs the `turkish-tokenizer` repo (`TURKISH_TOKENIZER_PATH`)
to segment each word into morphemes before applying the learned merges. A malformed/unreadable
`.json` falls back to `HFTokenCounter`.

```python
from turkish_corpus.tokenizer import load_token_counter

# turkish-llm exports → drop straight in:
load_token_counter("/models/byte_bpe_32000.json").count("evlerimizden")        # HF
load_token_counter("/models/sp_morph_32000.model").count("evlerimizden")       # SentencePiece
load_token_counter("/models/morpheme_bpe_20000.json").count("evlerimizden")    # MorphemeBPE
```

### Counting tokens *inside* the datatrove pipeline

The pipeline counts tokens with **any** of our counters, not just HF tokenizers. The final
stage dispatches on `config.TokenizerConfig.name_or_path` (CLI `--tokenizer`):

| `--tokenizer` spec | In-pipeline counter | Why |
|--------------------|---------------------|-----|
| HF `tokenizer.json` or Hub id | datatrove's native `TokensCounter` | fast, natively batched |
| `morpheme_bpe_<V>.json` (top-level `"morph_sep"`) | `TurkishTokensCounter` (→ `MorphemeBPETokenCounter`) | not exportable to HF; needs `tr_api` per word |
| SentencePiece `*.model` | `TurkishTokensCounter` (→ `SentencePieceTokenCounter`) | datatrove's counter can't load `.model` |

`TurkishTokensCounter` (in `turkish_corpus.blocks`) wraps the matching standalone
`TokenCounter`, records **`metadata["token_count"]` per document**, and emits a cumulative
`tokens` run-stat (plus `counted_docs`). It builds the counter lazily inside the datatrove
worker, so `tr_api`/HF import only happens where the work runs.

So the morpheme-aware BPE — turkish-llm's best tokenizer, which **cannot** be faithfully
exported to a single HF `tokenizer.json` because it needs the `tr_api` morphological analyzer
per word at inference (custom Python pre-tokenizers don't serialize) — is now countable
*in-pipeline* directly:

```bash
# Point the pipeline at the morpheme-BPE json and tell it where tr_api lives:
export TURKISH_TOKENIZER_PATH=/path/to/turkish-tokenizer
uv run --extra pipeline tc-run-hplt \
    --source jsonl --data-path /data/hplt/tur_Latn \
    --tokenizer /models/turkish-llm/morpheme_bpe_20000.json \
    --output /data/corpus/hplt_tur --tasks 64
```

**Speed caveat (morpheme-BPE only).** The morpheme-BPE counter runs the pure-Python `tr_api`
analyzer at ~600 words/s — too slow to count every document of a billion-token corpus. For
large runs set `TokenizerConfig.sample_rate < 1` to tokenize a *deterministic* sample (md5 of
`doc.id`, stable across runs/workers) and **estimate** the corpus total as
`tokens_stat × (1 / sample_rate)`. HF and SentencePiece counters are fast enough to leave
`sample_rate = 1.0` and count everything.

## Recommended flow

1. Train the morpheme-aware BPE (and baselines) in `turkish-llm`.
2. Point `tc-run-hplt --tokenizer` at the chosen export to get the real corpus token count —
   `morpheme_bpe_<V>.json` (set `TURKISH_TOKENIZER_PATH`), a SentencePiece `*.model`, or an HF
   `tokenizer.json` / Hub id all work; the pipeline picks the right counter automatically.
3. For the morpheme-BPE counter at corpus scale, set `sample_rate < 1` and multiply the
   `tokens` stat by `1 / sample_rate` to estimate the total (see the speed caveat above).
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

# Measure the morpheme-aware BPE (turkish-llm's best tokenizer) the same way — it needs the
# segmenter to analyse each word into morphemes before applying the learned merges:
uv run python scripts/measure_fertility.py \
    --tokenizer /models/morpheme_bpe_20000.json \
    --input sample_tr.txt \
    --turkish-tokenizer-path /path/to/turkish-tokenizer
```

Output: `tokens`, `words`, `tokens/word`, and (with the segmenter) `morphemes`,
`morphemes/word`, `subwords/morpheme`. Use it to pick the tokenizer with the lowest fertility
that still respects morpheme boundaries.

## API surface

- `turkish_corpus.tokenizer` — `TokenCounter` protocol, `WhitespaceTokenCounter`,
  `HFTokenCounter`, `SentencePieceTokenCounter`, `MorphemeBPE`, `MorphemeBPETokenCounter`
  (re-exported from `turkish_corpus.morpheme_bpe`), `load_token_counter`,
  `is_morpheme_bpe_spec(spec) -> bool` (public discriminator used by both the standalone
  loader and the in-pipeline dispatch).
- `turkish_corpus.blocks` — `TurkishTokensCounter` (in-pipeline counting with any
  `TokenCounter`; records `metadata["token_count"]` + a `tokens` stat; `sample_rate` for the
  slow morpheme-BPE counter).
- `turkish_corpus.morpheme_bpe` — `MorphemeBPE` (pure merge engine, `.from_file(path)`,
  `.encode(morphemes)`), `MorphemeBPETokenCounter` (`.encode_word(word)`, `.count(text)`).
- `turkish_corpus.morphology` — `ensure_tr_api_importable(repo_path=None)` (adds the
  `turkish-tokenizer` repo to `sys.path`, like turkish-llm's `_tok.py`; resolves from arg or
  `TURKISH_TOKENIZER_PATH`), `MorphologicalSegmenter` (`.segment(text)`, `.count(text)`).
- `turkish_corpus.fertility` — `compute_fertility(counter, texts)`,
  `compare_fertility(counter, texts, segmenter=...)`.
