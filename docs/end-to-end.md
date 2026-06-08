# End-to-end corpus build (one recipe, one command)

`turkish_corpus.orchestrate` runs the **whole** pipeline — ingest → clean → blend — from a
single declarative recipe file. It does not reimplement any stage; it sequences the existing
ones:

```
ingest  (turkish_corpus.sources.*)         raw JSONL per source
   -> clean   (turkish_corpus.pipeline.run_local)   cleaned JSONL  (.../final)
   -> blend   (turkish_corpus.blend.build_blend)     final register-diverse corpus
```

## One command

```bash
uv run --extra pipeline --extra sources --extra academic \
    tc-build-corpus --recipe examples/recipe.json
```

`tc-build-corpus` is the console entry (`turkish_corpus.orchestrate:main`). The loose-script
form is equivalent:

```bash
uv run --extra pipeline --extra sources --extra academic \
    python scripts/build_corpus.py --recipe examples/recipe.json
```

The extras map to the stages used: `pipeline` (datatrove cleaning), `sources` (`datasets`
for HPLT/Wikipedia/mevzuat), `academic` (PDF text extraction). A recipe that omits a stage
doesn't need its extra.

## Recipe schema

JSON or TOML (chosen by file extension). Top-level fields:

| Field | Type | Notes |
|-------|------|-------|
| `output_root` | str (required) | Root for all build artifacts. |
| `target_tokens` | int or str (required) | Total budget. Accepts `20e9` / `20_000_000_000` / `20000000000`. |
| `tokenizer` | str or null | Tokenizer spec for the in-pipeline counter and blend sizing. `null` → the whitespace counter (works before a tokenizer is trained). |
| `tasks` | int (default 4) | Local executor parallelism per cleaning run. |
| `shuffle_seed` | int (default 0) | Deterministic per-source shuffle for the blend. |
| `sources` | list (required, non-empty) | The steps to ingest/clean/blend. |

Each entry in `sources`:

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Unique id; drives `raw/<name>`, `clean/<name>`, and the manifest entry. |
| `kind` | str | One of `hplt`, `wikipedia`, `mevzuat`, `academic`, `jsonl`. |
| `weight` | float > 0 | Share of the budget; normalized across sources by the blend. |
| `license`, `register` | str (optional) | Source-level manifest labels. |

Kind-specific params (write them flat alongside the fields above, or nested under `params`):

| kind | ingest? | reader | params |
|------|---------|--------|--------|
| `hplt` | no (streams from HF Hub) | `hf` | `language` (default `tur_Latn`), `limit` |
| `wikipedia` | yes | `jsonl` | `dump` (default `20231101.tr`), `limit` |
| `mevzuat` | yes | `jsonl` | `limit` |
| `academic` | yes | `jsonl` | `pdf_dir` (**required**), `source` (`dergipark`\|`yoktez`, default `dergipark`), `limit` |
| `jsonl` | no (consumes a pre-produced raw dir) | `jsonl` | `raw_dir` (**required**) |

`limit` (`-1` = all) caps the HF/ingest read for smoke runs. It does not apply to `jsonl`
(an on-disk raw dir is read in full).

See `examples/recipe.json` for a realistic ~20B-token blend (hplt 0.60, academic 0.18,
wikipedia 0.12, mevzuat 0.10).

## Downloaders run separately

The `academic` and `jsonl` kinds consume directories produced by the **heavy downloaders /
scrapers**, which are intentionally a separate offline step (network-bound, rate-limited,
sometimes CAPTCHA-gated):

- `academic.pdf_dir` ← `scripts/download_dergipark.py` (DergiPark PDFs), YÖK portal, etc.
- `jsonl.raw_dir` ← `scripts/download_govlegal.py` / `scripts/run_crawl.py` and other
  scrapers that emit raw datatrove-shaped JSONL.

Produce those dirs first, then point the recipe at them. The orchestrator never downloads on
your behalf for these kinds.

## Output layout

```
<output_root>/
  raw/<name>/              raw JSONL from ingesters (wikipedia, mevzuat, academic)
  clean/<name>/            cleaning-pipeline output for <name>
  clean/<name>/final/      the cleaned result (what the blend reads)
  clean/<name>/logs/       datatrove logs
  blend/                   final blended shards, one <name>.jsonl.gz per source
  manifest.json            the blend manifest (from build_blend)
  corpus_manifest.json     recipe summary + blend manifest (the audit record)
```

`corpus_manifest.json` combines the build *intent* (budget, tokenizer, per-source
kind/weight/license/register) with the *achieved* numbers (per-source achieved/target
tokens, docs, shortfall, grand total) so the build is reproducible and auditable.

## Idempotency and `--force`

Cleaning is the expensive stage, so it is idempotent: if `clean/<name>/final` already exists
and is non-empty, that source is **skipped** on a re-run. This means adding one source to a
recipe (or recovering from a crash) only does the missing work. Pass `--force` to re-clean
every source from scratch.

## CLI overrides

`--recipe` is required; these override recipe fields without editing the file:

| Flag | Overrides |
|------|-----------|
| `--output-root` | `output_root` |
| `--target-tokens` | `target_tokens` (same `20e9` forms) |
| `--tokenizer` | `tokenizer` (empty string → whitespace counter) |
| `--tasks` | `tasks` |
| `--force` | re-clean all sources |

`--help` works without any extra installed (the heavy deps are imported lazily only when a
build actually runs).

## Sizing guidance

How big to make `target_tokens` (Chinchilla-optimal tokens per model scale, the role of the
HPLT web anchor vs. the curated registers) is covered in [blend.md](blend.md).
