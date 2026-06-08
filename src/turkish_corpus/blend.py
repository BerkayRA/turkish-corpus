"""Combine multiple CLEANED sources into the final register-diverse corpus at a budget.

This is the BLEND/MIXER stage — the last step of the roadmap. Each source has already
been ingested to raw JSONL and run through the datatrove cleaning pipeline
(:mod:`turkish_corpus.pipeline`), producing a directory of ``*.jsonl`` / ``*.jsonl.gz``
files whose lines are ``{"id","text","metadata":{source,license,register,...}}``. The
blend reads those cleaned dirs and mixes them so the final corpus hits a target *token*
budget with a chosen register mix (e.g. 12% encyclopedic, 30% legal, 40% web, …).

Why this exists separately from the cleaning pipeline
-----------------------------------------------------
- **Per-source dedup already happened** in the cleaning pipeline (the MinHash stages run
  within each source). The blend does NOT re-run that.
- **Cross-source dedup** (a document that appears in two different sources) is an OPTIONAL
  extra pass over the *union* of sources, exposed here as :func:`cross_source_dedup`. It is
  deliberately a separate concern: most blends don't need it, and it's expensive.
- **The blend itself** handles the three things the pipeline can't: weighting sources into
  a token budget, enforcing that budget per source with a real (pluggable) token counter,
  and emitting a manifest that audits achieved tokens / docs / licensing per source.

Token counting is pluggable via :func:`turkish_corpus.tokenizer.load_token_counter`:
``spec=None`` gives the dependency-free whitespace counter (so the blend works today,
before any tokenizer is trained), and a path / Hub id gives the real morpheme-aware BPE
once it's exported. Sizing must use the *same* counter the corpus is reported in, because
Turkish's tokenization premium makes whitespace-word counts and subword counts diverge a
lot (see :mod:`turkish_corpus.tokenizer`).

Memory / selection tradeoff
---------------------------
To avoid always taking the first N documents of a shard (which would bias the blend toward
whatever happens to sort first), selection is deterministically shuffled with
``random.Random(shuffle_seed)``. Holding every document's *text* in memory would be
prohibitive at billions of tokens, so :func:`build_blend` uses a **two-pass, index-only**
strategy: pass 1 streams each source and records only ``(shard_path, line_no, token_count)``
per doc (small, ~tens of bytes/doc), shuffles those lightweight index entries, and greedily
picks until the source's token target is met; pass 2 re-streams the shards and writes only
the selected lines. Memory is therefore O(total docs) in cheap index tuples, not O(corpus
text) — the right tradeoff for this scale. Token counts are computed once (pass 1) and not
recomputed in pass 2.

Pure stdlib (json, gzip, glob, os, random); the token counter is the only heavy import and
is loaded lazily.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "BlendSource",
    "iter_jsonl",
    "build_blend",
    "cross_source_dedup",
]


@dataclass(frozen=True)
class BlendSource:
    """One cleaned source to mix into the blend.

    Parameters
    ----------
    name:
        Short id for the source (e.g. ``"wikipedia"``), used in the manifest.
    path:
        Directory of CLEANED JSONL (``*.jsonl`` / ``*.jsonl.gz``) for this source — the
        ``final`` output dir of the cleaning pipeline.
    weight:
        Desired share of the target token budget. Weights across sources are normalized to
        sum 1.0, so relative proportions are what matter (``0.12`` vs ``0.30`` etc.).
    license:
        Optional license string for the manifest. Records also carry license in metadata;
        this is the source-level summary value.
    register:
        Optional register label (see :data:`turkish_corpus.sources.base.REGISTERS`).
    """

    name: str
    path: str
    weight: float
    license: str = ""
    register: str = ""


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield parsed records from every ``*.jsonl`` / ``*.jsonl.gz`` file under ``path``.

    Files are visited in sorted order for determinism. ``.gz`` files are read through gzip;
    plain ``.jsonl`` through the builtin opener. Blank lines are skipped. Pure stdlib so it
    imports and runs without any optional extra.
    """
    for shard in _shards(path):
        opener = gzip.open if shard.endswith(".gz") else open
        with opener(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)


def _shards(path: str) -> list[str]:
    """Return the sorted list of jsonl shard paths under ``path`` (gz and plain)."""
    plain = glob.glob(os.path.join(path, "*.jsonl"))
    gz = glob.glob(os.path.join(path, "*.jsonl.gz"))
    return sorted(plain + gz)


def _normalize_weights(sources: list[BlendSource]) -> list[float]:
    """Normalize source weights to sum 1.0; fail fast on non-positive total."""
    total = sum(s.weight for s in sources)
    if total <= 0:
        raise ValueError(f"source weights must sum to a positive number, got {total}")
    return [s.weight / total for s in sources]


@dataclass(frozen=True)
class _DocIndex:
    """A lightweight pointer to one document, with its precomputed token count.

    Holds no text — just enough to (a) decide selection and (b) find the line again in
    pass 2. This is the unit kept in memory (O(docs) cheap tuples, not O(text)).
    """

    shard: str
    line_no: int
    tokens: int


def _index_source(path: str, counter) -> tuple[list[_DocIndex], int]:
    """Pass 1: stream a source, returning per-doc index entries and total tokens.

    Token counts are computed here once with ``counter`` and carried on the index so pass 2
    never re-tokenizes. Records are located by ``(shard, line_no)`` so pass 2 can re-read
    exactly the chosen lines without holding their text.
    """
    index: list[_DocIndex] = []
    total = 0
    for shard in _shards(path):
        opener = gzip.open if shard.endswith(".gz") else open
        with opener(shard, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if not line.strip():
                    continue
                rec = json.loads(line)
                tokens = counter.count(rec.get("text") or "")
                index.append(_DocIndex(shard=shard, line_no=line_no, tokens=tokens))
                total += tokens
    return index, total


def _select(index: list[_DocIndex], target_tokens: int, rng: random.Random) -> set[tuple[str, int]]:
    """Greedily pick shuffled docs until ``target_tokens`` is reached (doc granularity).

    Returns the set of ``(shard, line_no)`` keys to keep. The shuffle is deterministic in
    ``rng`` so the same seed always yields the same blend. Selection stops on the first doc
    that meets-or-exceeds the target, so the achieved count can overshoot by at most one
    document — accepted, and reported as exact achieved tokens in the manifest.
    """
    order = list(index)
    rng.shuffle(order)
    chosen: set[tuple[str, int]] = set()
    acc = 0
    for doc in order:
        if acc >= target_tokens:
            break
        chosen.add((doc.shard, doc.line_no))
        acc += doc.tokens
    return chosen


def _write_selected(path: str, selected: set[tuple[str, int]], out_fh) -> int:
    """Pass 2: re-stream a source's shards, writing only selected lines; return doc count.

    Tokens are NOT recomputed here — the caller already has exact per-doc counts from pass 1.
    Lines are re-emitted verbatim (they're already cleaned), so no text is held in memory.
    """
    docs = 0
    for shard in _shards(path):
        opener = gzip.open if shard.endswith(".gz") else open
        with opener(shard, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if (shard, line_no) not in selected:
                    continue
                out_fh.write(line if line.endswith("\n") else line + "\n")
                docs += 1
    return docs


def build_blend(
    sources: list[BlendSource],
    out_dir: str,
    *,
    target_tokens: int,
    tokenizer_spec: str | None = None,
    shuffle_seed: int = 0,
) -> dict:
    """Mix ``sources`` into ``out_dir/blend/`` at ``target_tokens``, returning a manifest.

    Each source gets a token target of ``round(normalized_weight * target_tokens)``. For
    each source we stream-index its docs (counting tokens with the configured counter),
    deterministically shuffle, and greedily select docs until the source's target is met or
    the source is exhausted (recording the shortfall). Selected docs are written to
    ``out_dir/blend/`` as one gzipped JSONL shard per source (``<name>.jsonl.gz``).

    See the module docstring for the memory/selection tradeoff (two-pass, index-only).

    Parameters
    ----------
    sources:
        The cleaned sources to mix. Weights are normalized to sum 1.0.
    out_dir:
        Output root. The blended shards go in ``out_dir/blend/``; the manifest in
        ``out_dir/manifest.json``.
    target_tokens:
        Total token budget for the blended corpus, measured in the configured counter.
    tokenizer_spec:
        Passed to :func:`turkish_corpus.tokenizer.load_token_counter`. ``None`` -> the
        whitespace counter (works with no trained tokenizer); a path / Hub id -> the real
        tokenizer.
    shuffle_seed:
        Seed for deterministic per-source document shuffling.

    Returns
    -------
    dict
        The manifest (also written to ``out_dir/manifest.json``): per-source entries and a
        ``totals`` block. See :func:`_build_manifest` for the exact shape.
    """
    if not sources:
        raise ValueError("build_blend requires at least one source")
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")

    # Lazy heavy import: only load the token counter when actually blending.
    from .tokenizer import load_token_counter  # noqa: PLC0415

    counter = load_token_counter(tokenizer_spec)
    tokenizer_name = getattr(counter, "name", tokenizer_spec or "whitespace")

    weights = _normalize_weights(sources)
    blend_dir = os.path.join(out_dir, "blend")
    os.makedirs(blend_dir, exist_ok=True)

    per_source: list[dict] = []
    for src, weight in zip(sources, weights, strict=True):
        src_target = round(weight * target_tokens)
        index, available = _index_source(src.path, counter)

        rng = random.Random(shuffle_seed)
        selected = _select(index, src_target, rng)
        # Achieved tokens come from the pass-1 counts of selected docs — no re-tokenizing.
        achieved = sum(d.tokens for d in index if (d.shard, d.line_no) in selected)

        shard_path = os.path.join(blend_dir, f"{src.name}.jsonl.gz")
        with gzip.open(shard_path, "wt", encoding="utf-8") as out_fh:
            docs = _write_selected(src.path, selected, out_fh)

        shortfall = max(0, src_target - achieved)
        per_source.append(
            {
                "name": src.name,
                "license": src.license,
                "register": src.register,
                "weight": weight,
                "target_tokens": src_target,
                "achieved_tokens": achieved,
                "docs": docs,
                "shortfall_tokens": shortfall,
                "available_tokens": available,
            }
        )

    manifest = _build_manifest(per_source, target_tokens, tokenizer_name)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def _build_manifest(per_source: list[dict], target_tokens: int, tokenizer_name: str) -> dict:
    """Assemble the manifest dict from per-source entries plus rolled-up totals."""
    return {
        "sources": per_source,
        "totals": {
            "target_tokens": target_tokens,
            "achieved_tokens": sum(s["achieved_tokens"] for s in per_source),
            "docs": sum(s["docs"] for s in per_source),
            "shortfall_tokens": sum(s["shortfall_tokens"] for s in per_source),
            "tokenizer": tokenizer_name,
        },
    }


def cross_source_dedup(
    source_dirs: list[str],
    out_dir: str,
    *,
    config=None,
    tasks: int = 4,
    logging_dir: str = "./logs/blend_xdedup",
):
    """OPTIONAL extra: MinHash near-dedup across the UNION of cleaned sources, before blending.

    Per-source dedup already ran inside the cleaning pipeline; this catches documents that
    are near-duplicates *across* sources (e.g. a Wikipedia article mirrored on a web crawl).
    Run it BEFORE :func:`build_blend`, pointing the blend's :class:`BlendSource` paths at the
    deduped output instead of the raw cleaned dirs.

    This reuses the exact 4-stage MinHash topology from :func:`turkish_corpus.pipeline.run_local`
    (signatures -> buckets -> cluster -> filter), but reads the union of ``source_dirs`` so
    duplicates are detected globally. datatrove is a lazy import behind the ``pipeline`` extra.

    Parameters
    ----------
    source_dirs:
        Cleaned source directories to dedup against each other.
    out_dir:
        Root for the deduped output (``out_dir/deduped``) and MinHash intermediates.
    config:
        A :class:`turkish_corpus.config.PipelineConfig`; its ``minhash`` params drive the
        signature stage. Defaults to :func:`turkish_corpus.config.default_hplt_config`.
    tasks:
        Local executor parallelism for the read/sig/filter stages.
    logging_dir:
        datatrove logging directory.

    Returns
    -------
    The final datatrove executor (already run).
    """
    try:
        import datatrove  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
        raise ImportError(
            "cross_source_dedup requires datatrove. Install with `uv sync --extra pipeline`."
        ) from exc

    from datatrove.executor import LocalPipelineExecutor  # noqa: PLC0415
    from datatrove.pipeline.dedup.minhash import (  # noqa: PLC0415
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.readers import JsonlReader  # noqa: PLC0415
    from datatrove.pipeline.writers import JsonlWriter  # noqa: PLC0415
    from datatrove.utils.hashing import HashConfig  # noqa: PLC0415
    from datatrove.utils.typeshelper import Languages  # noqa: PLC0415

    from .config import default_hplt_config  # noqa: PLC0415

    cfg = config or default_hplt_config()
    m = cfg.minhash
    mh = MinhashConfig(
        hash_config=HashConfig(precision=m.precision),
        num_buckets=m.num_buckets,
        hashes_per_bucket=m.hashes_per_bucket,
        n_grams=m.n_grams,
    )

    def _p(sub: str) -> str:
        return os.path.join(out_dir, sub)

    def _readers() -> list:
        # One reader per source dir; their records flow into a single MinHash run, so
        # signatures are computed over the union and duplicates are found cross-source.
        return [JsonlReader(d) for d in source_dirs]

    sig = LocalPipelineExecutor(
        pipeline=[
            *_readers(),
            MinhashDedupSignature(
                output_folder=_p("minhash/signatures"),
                config=mh,
                language=Languages.turkish,
            ),
        ],
        tasks=tasks,
        logging_dir=os.path.join(logging_dir, "sig"),
    )
    buckets = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=_p("minhash/signatures"),
                output_folder=_p("minhash/buckets"),
                config=mh,
            )
        ],
        tasks=mh.num_buckets,
        logging_dir=os.path.join(logging_dir, "buckets"),
        depends=sig,
    )
    cluster = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=_p("minhash/buckets"),
                output_folder=_p("minhash/remove_ids"),
                config=mh,
            )
        ],
        tasks=1,
        logging_dir=os.path.join(logging_dir, "cluster"),
        depends=buckets,
    )
    final = LocalPipelineExecutor(
        pipeline=[
            *_readers(),
            MinhashDedupFilter(
                input_folder=_p("minhash/remove_ids"),
                exclusion_writer=JsonlWriter(_p("removed_cross_duplicates")),
            ),
            JsonlWriter(output_folder=_p("deduped")),
        ],
        tasks=tasks,
        logging_dir=os.path.join(logging_dir, "final"),
        depends=cluster,
    )
    final.run()
    return final
