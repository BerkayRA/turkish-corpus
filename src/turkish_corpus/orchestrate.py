"""End-to-end corpus orchestrator: ingest -> clean -> blend from one recipe, one command.

This is the top of the roadmap stack. Each lower layer already exists and is reused
verbatim — this module only *sequences* them per a declarative recipe:

    ingest  (turkish_corpus.sources.*)   raw JSONL per source
       -> clean   (turkish_corpus.pipeline.run_local)   cleaned JSONL  (.../final)
       -> blend   (turkish_corpus.blend.build_blend)     final register-diverse corpus

A *recipe* (JSON or TOML) lists the sources, their blend weights, and the global token
budget; :func:`run_corpus` walks it, materializing each layer under one ``output_root`` and
emitting a top-level ``corpus_manifest.json`` that combines the recipe summary with the
blend manifest (per-source achieved/target tokens, licenses, registers).

Why a recipe rather than flags
------------------------------
The full pipeline has many sources, each with its own ingest knobs, plus weights and a
budget. A recipe makes the whole build reproducible, reviewable, and diffable in one file,
and keeps the heavy downloaders (DergiPark PDFs, gov scrapers) as a *separate* offline step
whose output dirs are simply referenced here (``academic.pdf_dir`` / ``jsonl.raw_dir``).

Idempotency
-----------
Cleaning is the expensive stage. If ``clean/<name>/final`` already exists and is non-empty,
the step is skipped unless ``force=True`` — so re-running after adding one source, or after a
crash, only does the missing work.

Heavy imports (datatrove via :func:`turkish_corpus.pipeline.run_local`, ``datasets`` via the
ingesters, the token counter via :func:`turkish_corpus.blend.build_blend`) are all loaded
LAZILY inside :func:`run_corpus`, so this module — and ``--help`` — work without any extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from dataclasses import dataclass, field

from .config import default_hplt_config

__all__ = [
    "CorpusStep",
    "CorpusRecipe",
    "SUPPORTED_KINDS",
    "load_recipe",
    "run_corpus",
    "main",
]

logger = logging.getLogger("turkish_corpus.orchestrate")

# The step kinds the orchestrator knows how to materialize. Kept as a frozenset so recipe
# validation can give an exact "unknown kind" error listing the supported set.
SUPPORTED_KINDS = frozenset({"hplt", "wikipedia", "mevzuat", "academic", "jsonl"})

# Default HPLT language; matches turkish_corpus.config.HPLT_LANG. Spelled out here so a
# recipe can omit ``params.language`` for the common case.
_DEFAULT_HPLT_LANGUAGE = "tur_Latn"


@dataclass
class CorpusStep:
    """One source in the recipe: how to produce it and how heavily to blend it.

    Parameters
    ----------
    name:
        Short, unique id for this step. Drives the on-disk layout (``raw/<name>``,
        ``clean/<name>``) and the blend manifest entry, so it must be unique within a recipe.
    kind:
        One of :data:`SUPPORTED_KINDS`. Selects the ingest path (or none) and the reader:

        - ``"hplt"`` — clean HPLT v2 directly from the HF Hub (no ingest). ``params``:
          ``language`` (default ``"tur_Latn"``), ``limit``.
        - ``"wikipedia"`` — ingest Turkish Wikipedia. ``params``: ``dump``, ``limit``.
        - ``"mevzuat"`` — ingest the mevzuat.gov.tr HF mirror. ``params``: ``limit``.
        - ``"academic"`` — ingest pre-downloaded academic PDFs. ``params``: ``pdf_dir``
          (required), ``source`` (``"dergipark"`` | ``"yoktez"``, default ``"dergipark"``),
          ``limit``.
        - ``"jsonl"`` — clean a pre-produced raw JSONL dir (from a downloader/scraper run
          separately). ``params``: ``raw_dir`` (required). No ingest.
    weight:
        Desired share of the token budget; normalized across steps by the blend. Must be > 0.
    license, register:
        Optional manifest labels. When omitted, ingesters still stamp provenance into each
        record's metadata; these are the source-level summary values for the corpus manifest.
    params:
        Kind-specific options (see ``kind`` above).
    """

    name: str
    kind: str
    weight: float
    license: str = ""
    register: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class CorpusRecipe:
    """A full corpus build: the sources to blend, the budget, and global run settings.

    Parameters
    ----------
    sources:
        The ordered list of :class:`CorpusStep` to ingest/clean/blend. Must be non-empty.
    output_root:
        Root directory for the whole build. Layout: ``raw/<name>/``, ``clean/<name>/``
        (with ``clean/<name>/final`` the cleaned result), ``blend/``, and the top-level
        ``corpus_manifest.json``.
    target_tokens:
        Total token budget for the blended corpus, measured in ``tokenizer``'s counter.
    tokenizer:
        Tokenizer spec passed to both the in-pipeline token counter and the blend's sizing
        counter. ``None`` -> the dependency-free whitespace counter (works before a tokenizer
        is trained). See :mod:`turkish_corpus.tokenizer`.
    tasks:
        Local executor parallelism for each cleaning run.
    shuffle_seed:
        Deterministic per-source shuffle seed for the blend.
    """

    sources: list[CorpusStep]
    output_root: str
    target_tokens: int
    tokenizer: str | None = None
    tasks: int = 4
    shuffle_seed: int = 0

    def validate(self) -> None:
        """Fail fast on a malformed recipe before any expensive work starts."""
        if not self.sources:
            raise ValueError("recipe must list at least one source")
        if self.target_tokens <= 0:
            raise ValueError(f"target_tokens must be positive, got {self.target_tokens}")

        names = [s.name for s in self.sources]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"source names must be unique; duplicated: {duplicates}")

        for step in self.sources:
            if step.kind not in SUPPORTED_KINDS:
                raise ValueError(
                    f"source {step.name!r}: unknown kind {step.kind!r}; "
                    f"supported: {sorted(SUPPORTED_KINDS)}"
                )
            if step.weight <= 0:
                raise ValueError(
                    f"source {step.name!r}: weight must be positive, got {step.weight}"
                )
            # Required params are kind-specific: academic needs the PDF dir to extract,
            # jsonl needs the pre-produced raw dir to clean. Check them up front so the
            # operator isn't told mid-build (after some sources already cleaned).
            if step.kind == "academic" and not step.params.get("pdf_dir"):
                raise ValueError(f"academic source {step.name!r} requires params.pdf_dir")
            if step.kind == "jsonl" and not step.params.get("raw_dir"):
                raise ValueError(f"jsonl source {step.name!r} requires params.raw_dir")


def _parse_token_budget(value: int | str) -> int:
    """Parse a token budget given as int or string (``"20e9"``, ``"20_000_000_000"``) to int.

    Mirrors ``scripts/build_blend.py``: a plain integer string is parsed as ``int`` first to
    avoid float precision loss on huge values (``int(float(...))`` rounds past 2**53), and
    only scientific-notation forms fall back to the float path. An ``int`` passes through.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        budget = value
    else:
        cleaned = str(value).replace("_", "").strip()
        try:
            budget = int(cleaned)
        except ValueError:
            try:
                budget = int(float(cleaned))
            except ValueError as exc:
                raise ValueError(f"target_tokens must be a number, got {value!r}") from exc
    if budget <= 0:
        raise ValueError(f"target_tokens must be positive, got {value!r}")
    return budget


def _step_from_dict(raw: dict) -> CorpusStep:
    """Build one :class:`CorpusStep` from a recipe dict entry, copying unknown keys to params.

    ``name``/``kind``/``weight``/``license``/``register`` are top-level fields; everything
    else (``language``, ``dump``, ``limit``, ``pdf_dir``, ``raw_dir``, ``source``, …) is
    collected under ``params`` so the recipe schema stays flat and readable.
    """
    known = {"name", "kind", "weight", "license", "register", "params"}
    if "name" not in raw or "kind" not in raw or "weight" not in raw:
        raise ValueError(f"each source needs name/kind/weight; got keys {sorted(raw)}")
    # Allow params either nested under "params" or flattened alongside the known fields.
    params = dict(raw.get("params") or {})
    for key, value in raw.items():
        if key not in known:
            params[key] = value
    return CorpusStep(
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        weight=float(raw["weight"]),
        license=str(raw.get("license", "")),
        register=str(raw.get("register", "")),
        params=params,
    )


def load_recipe(path: str) -> CorpusRecipe:
    """Load and validate a recipe from a JSON or TOML file (chosen by extension).

    ``.toml`` is read in binary via :mod:`tomllib`; anything else is parsed as JSON.
    ``target_tokens`` may be an int or a string (``"20e9"`` / ``"20_000_000_000"``). The
    returned recipe is already ``validate()``-d, so callers can run it directly.
    """
    if path.endswith(".toml"):
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

    if "sources" not in data:
        raise ValueError("recipe must have a 'sources' list")
    if "output_root" not in data:
        raise ValueError("recipe must set 'output_root'")
    if "target_tokens" not in data:
        raise ValueError("recipe must set 'target_tokens'")

    recipe = CorpusRecipe(
        sources=[_step_from_dict(s) for s in data["sources"]],
        output_root=str(data["output_root"]),
        target_tokens=_parse_token_budget(data["target_tokens"]),
        tokenizer=data.get("tokenizer"),
        tasks=int(data.get("tasks", 4)),
        shuffle_seed=int(data.get("shuffle_seed", 0)),
    )
    recipe.validate()
    return recipe


def _final_dir_is_populated(final_dir: str) -> bool:
    """Return True if ``final_dir`` exists and holds at least one file (idempotency probe)."""
    return os.path.isdir(final_dir) and any(os.scandir(final_dir))


def _ingest_step(step: CorpusStep, raw_dir: str) -> int:
    """Run the ingester for an ingest-kind step into ``raw_dir``; return records written.

    Ingesters are imported LAZILY (they pull ``datasets`` / PDF deps behind the ``sources`` /
    ``academic`` extras) so this module imports without them. ``hplt`` and ``jsonl`` never
    reach here — they have no ingest stage.
    """
    p = step.params
    limit = int(p.get("limit", -1))
    if step.kind == "wikipedia":
        from .sources.wikipedia import ingest_wikipedia  # noqa: PLC0415

        return ingest_wikipedia(raw_dir, dump=p.get("dump", "20231101.tr"), limit=limit)
    if step.kind == "mevzuat":
        from .sources.govlegal import ingest_mevzuat  # noqa: PLC0415

        return ingest_mevzuat(raw_dir, limit=limit)
    if step.kind == "academic":
        from .sources.academic import ingest_dergipark, ingest_yoktez  # noqa: PLC0415

        source = p.get("source", "dergipark")
        ingest = ingest_dergipark if source == "dergipark" else ingest_yoktez
        return ingest(raw_dir, pdf_dir=p["pdf_dir"], limit=limit)
    raise ValueError(f"kind {step.kind!r} has no ingest stage")


def _config_for_step(step: CorpusStep, recipe: CorpusRecipe, raw_dir: str, clean_dir: str):
    """Build the :class:`PipelineConfig` that cleans ``step`` into ``clean_dir``.

    ``hplt`` reads from the HF Hub (``source="hf"``, the dataset's language config); every
    other kind reads the raw JSONL dir (``source="jsonl"``, ``data_path`` = the ingested
    ``raw_dir`` for ingest kinds, or the recipe's ``raw_dir`` for the ``jsonl`` kind). The
    ``limit`` only applies to HF/ingest reads (already-on-disk dirs are read in full).
    """
    p = step.params
    cfg = default_hplt_config()
    if step.kind == "hplt":
        cfg.reader.source = "hf"
        cfg.reader.language = p.get("language", _DEFAULT_HPLT_LANGUAGE)
        cfg.reader.limit = int(p.get("limit", -1))
    else:
        cfg.reader.source = "jsonl"
        cfg.reader.data_path = p["raw_dir"] if step.kind == "jsonl" else raw_dir
        # jsonl reads the whole dir; ingest kinds already applied limit during ingest.
        cfg.reader.limit = -1

    cfg.output_path = clean_dir
    cfg.logging_dir = os.path.join(clean_dir, "logs")
    cfg.tasks = recipe.tasks
    cfg.tokenizer.name_or_path = recipe.tokenizer
    cfg.validate()
    return cfg


def _clean_step(
    step: CorpusStep, recipe: CorpusRecipe, raw_dir: str, clean_dir: str, *, force: bool
) -> str:
    """Clean one step into ``clean_dir`` (skipping if already done); return its ``final`` dir.

    Idempotency: a populated ``clean_dir/final`` short-circuits unless ``force``. The
    datatrove run (:func:`turkish_corpus.pipeline.run_local`) is imported lazily behind the
    ``pipeline`` extra, so importing this module needs nothing.
    """
    final_dir = os.path.join(clean_dir, "final")
    if not force and _final_dir_is_populated(final_dir):
        logger.info("[%s] clean: skip (exists) %s", step.name, final_dir)
        return final_dir

    from .pipeline import run_local  # noqa: PLC0415

    config = _config_for_step(step, recipe, raw_dir, clean_dir)
    logger.info(
        "[%s] clean: reader=%s -> %s", step.name, config.reader.source, final_dir
    )
    run_local(config)
    return final_dir


def run_corpus(recipe: CorpusRecipe, *, force: bool = False) -> dict:
    """Run the whole pipeline for ``recipe`` and return the combined corpus manifest.

    For each source, in order: (1) produce its raw dir — ingest for wikipedia/mevzuat/
    academic, none for hplt/jsonl; (2) clean it with the existing datatrove pipeline into
    ``clean/<name>/final`` (skipped if already populated and not ``force``); (3) register it
    as a :class:`turkish_corpus.blend.BlendSource` pointing at that ``final`` dir. Then blend
    all sources to ``output_root/blend/`` at ``recipe.target_tokens`` and write
    ``output_root/corpus_manifest.json`` (recipe summary + blend manifest).

    Returns the corpus manifest dict (also written to disk). Heavy deps are imported lazily.
    """
    recipe.validate()
    from .blend import BlendSource, build_blend  # noqa: PLC0415

    root = recipe.output_root
    os.makedirs(root, exist_ok=True)
    logger.info(
        "build start: %d source(s), target=%d tokens, tokenizer=%s, root=%s",
        len(recipe.sources),
        recipe.target_tokens,
        recipe.tokenizer or "whitespace",
        root,
    )

    blend_sources: list[BlendSource] = []
    for step in recipe.sources:
        raw_dir = os.path.join(root, "raw", step.name)
        clean_dir = os.path.join(root, "clean", step.name)

        # 1) Raw dir. hplt streams from the Hub; jsonl points at a separately produced dir;
        # the rest ingest into raw/<name>.
        if step.kind in {"wikipedia", "mevzuat", "academic"}:
            count = _ingest_step(step, raw_dir)
            logger.info("[%s] ingest: wrote %d record(s) to %s", step.name, count, raw_dir)
        elif step.kind == "jsonl":
            logger.info("[%s] ingest: skip (raw_dir=%s)", step.name, step.params["raw_dir"])
        else:  # hplt
            logger.info("[%s] ingest: skip (HPLT streamed from HF)", step.name)

        # 2) Clean.
        final_dir = _clean_step(step, recipe, raw_dir, clean_dir, force=force)

        # 3) Register for the blend.
        blend_sources.append(
            BlendSource(
                name=step.name,
                path=final_dir,
                weight=step.weight,
                license=step.license,
                register=step.register,
            )
        )

    blend_dir = os.path.join(root, "blend")
    logger.info("blend: mixing %d source(s) into %s", len(blend_sources), blend_dir)
    blend_manifest = build_blend(
        blend_sources,
        root,
        target_tokens=recipe.target_tokens,
        tokenizer_spec=recipe.tokenizer,
        shuffle_seed=recipe.shuffle_seed,
    )

    manifest = _build_corpus_manifest(recipe, blend_manifest)
    manifest_path = os.path.join(root, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    totals = blend_manifest["totals"]
    logger.info(
        "build done: %d/%d tokens across %d docs -> %s",
        totals["achieved_tokens"],
        totals["target_tokens"],
        totals["docs"],
        manifest_path,
    )
    return manifest


def _build_corpus_manifest(recipe: CorpusRecipe, blend_manifest: dict) -> dict:
    """Combine the recipe summary with the blend manifest into the top-level corpus manifest.

    The ``recipe`` block captures the build's intent (budget, tokenizer, per-source kind/
    weight/license/register); the ``blend`` block carries the achieved numbers from
    :func:`turkish_corpus.blend.build_blend`. Together they make the build auditable.
    """
    return {
        "recipe": {
            "output_root": recipe.output_root,
            "target_tokens": recipe.target_tokens,
            "tokenizer": recipe.tokenizer,
            "tasks": recipe.tasks,
            "shuffle_seed": recipe.shuffle_seed,
            "sources": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "weight": s.weight,
                    "license": s.license,
                    "register": s.register,
                    "params": s.params,
                }
                for s in recipe.sources
            ],
        },
        "blend": blend_manifest,
    }


def _apply_overrides(recipe: CorpusRecipe, args: argparse.Namespace) -> CorpusRecipe:
    """Return a recipe with CLI overrides applied (immutably building a new recipe).

    Each ``--`` override (output-root, target-tokens, tokenizer, tasks) replaces the recipe
    value when provided; ``target_tokens`` reuses :func:`_parse_token_budget` so the CLI
    accepts the same ``"20e9"`` forms as the recipe file. ``--tokenizer ""`` clears it (back
    to the whitespace counter). The result is re-validated.
    """
    target = (
        _parse_token_budget(args.target_tokens)
        if args.target_tokens is not None
        else recipe.target_tokens
    )
    tokenizer = recipe.tokenizer
    if args.tokenizer is not None:
        tokenizer = args.tokenizer or None  # empty string -> whitespace counter

    updated = CorpusRecipe(
        sources=recipe.sources,
        output_root=args.output_root or recipe.output_root,
        target_tokens=target,
        tokenizer=tokenizer,
        tasks=args.tasks if args.tasks is not None else recipe.tasks,
        shuffle_seed=recipe.shuffle_seed,
    )
    updated.validate()
    return updated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tc-build-corpus",
        description="Run the whole corpus pipeline (ingest -> clean -> blend) from one recipe.",
    )
    p.add_argument("--recipe", required=True, help="Path to a recipe file (.json or .toml).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-clean every source even if its clean/<name>/final already exists.",
    )
    p.add_argument("--output-root", default=None, help="Override the recipe's output_root.")
    p.add_argument(
        "--target-tokens",
        default=None,
        help="Override the recipe's token budget (accepts 20e9 / 20_000_000_000).",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Override the tokenizer spec. Pass an empty string for the whitespace counter.",
    )
    p.add_argument("--tasks", type=int, default=None, help="Override local executor parallelism.")
    return p


def _print_summary(manifest: dict) -> None:
    """Print a per-source achieved/target token summary plus the grand total and output path."""
    recipe = manifest["recipe"]
    blend = manifest["blend"]
    totals = blend["totals"]
    print(f"tokenizer:     {totals['tokenizer']}")
    print(f"{'source':<16}{'achieved':>16}{'target':>16}{'docs':>12}{'shortfall':>16}")
    for s in blend["sources"]:
        print(
            f"{s['name']:<16}{s['achieved_tokens']:>16,}{s['target_tokens']:>16,}"
            f"{s['docs']:>12,}{s['shortfall_tokens']:>16,}"
        )
    print(
        f"{'TOTAL':<16}{totals['achieved_tokens']:>16,}{totals['target_tokens']:>16,}"
        f"{totals['docs']:>12,}{totals['shortfall_tokens']:>16,}"
    )
    root = recipe["output_root"]
    print(f"\nblend shards:    {os.path.join(root, 'blend')}")
    print(f"corpus manifest: {os.path.join(root, 'corpus_manifest.json')}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Configure logging only at the CLI boundary so library callers keep their own handlers.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    recipe = load_recipe(args.recipe)
    recipe = _apply_overrides(recipe, args)
    manifest = run_corpus(recipe, force=args.force)
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
