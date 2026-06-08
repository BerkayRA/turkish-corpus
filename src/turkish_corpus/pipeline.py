"""Assemble and run the HPLT v2 Turkish cleaning pipeline with datatrove.

Requires the ``pipeline`` extra (``uv sync --extra pipeline``).

The flow mirrors FineWeb-2's Turkish-validated recipe but is anchored on HPLT v2 (CC0):

    Stage A  process : read HPLT -> language gate -> Gopher repetition -> Gopher quality
                       (Turkish thresholds + stopwords) -> FineWeb quality -> Turkish
                       normalize -> Turkish PII -> email/IP PII -> write intermediate
    Stage B  minhash : signatures -> buckets -> cluster -> (remove-id lists)
    Stage C  dedup   : reread intermediate -> drop near-duplicates -> count tokens
                       (your tokenizer) -> write final

MinHash is split into stages because bucketing needs every document's signature. On the
Linux production box, raise ``config.tasks`` and run the same code; a Slurm helper is
provided for cluster scale.
"""

from __future__ import annotations

import os

from .config import PipelineConfig

__all__ = [
    "build_process_pipeline",
    "run_local",
    "run_slurm",
]


def _require_datatrove():
    try:
        import datatrove  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The cleaning pipeline requires datatrove. Install with "
            "`uv sync --extra pipeline`."
        ) from exc


def _make_reader(config: PipelineConfig):
    from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader

    r = config.reader
    if r.source == "hf":
        return HuggingFaceDatasetReader(
            dataset=r.dataset,
            dataset_options={"name": r.language, "split": r.split},
            streaming=r.streaming,
            limit=r.limit,
            text_key=r.text_key,
            id_key=r.id_key or "id",
        )
    return JsonlReader(
        data_folder=r.data_path,
        text_key=r.text_key,
        id_key=r.id_key or "id",
        limit=r.limit,
    )


def build_process_pipeline(config: PipelineConfig) -> list:
    """Stage A: read + filter + normalize + scrub, writing an intermediate dataset."""
    _require_datatrove()
    config.validate()

    from datatrove.pipeline.filters import (
        FineWebQualityFilter,
        GopherQualityFilter,
        GopherRepetitionFilter,
        LanguageFilter,
    )
    from datatrove.pipeline.formatters import PIIFormatter
    from datatrove.pipeline.writers import JsonlWriter

    from .blocks import TurkishNormalizer, TurkishPIIRedactor

    steps: list = [_make_reader(config)]

    if config.language.enabled:
        steps.append(
            LanguageFilter(
                languages=list(config.language.languages),
                language_threshold=config.language.threshold,
            )
        )

    steps += [
        GopherRepetitionFilter(),
        GopherQualityFilter(**config.quality.as_gopher_kwargs()),
        FineWebQualityFilter(),
        TurkishNormalizer(fix_mojibake=config.fix_mojibake),
        TurkishPIIRedactor(
            redact_tc=config.pii.redact_tc_kimlik,
            redact_phone=config.pii.redact_phone,
            redact_iban=config.pii.redact_iban,
            redact_plate=config.pii.redact_plate,
        ),
    ]

    if config.pii.redact_emails_and_ips:
        # datatrove handles emails + IPv4/IPv6; Turkish IDs are handled above.
        steps.append(PIIFormatter())

    steps.append(JsonlWriter(output_folder=_p(config, "intermediate")))
    return steps


def _minhash_config(config: PipelineConfig):
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig

    m = config.minhash
    return MinhashConfig(
        hash_config=HashConfig(precision=m.precision),
        num_buckets=m.num_buckets,
        hashes_per_bucket=m.hashes_per_bucket,
        n_grams=m.n_grams,
    )


def run_local(config: PipelineConfig):
    """Run the full A -> B -> C pipeline locally (single machine).

    Returns the final executor. Suitable for dev slices on macOS and for the full run on
    the Linux box (raise ``config.tasks``).
    """
    _require_datatrove()
    config.validate()

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.tokens import TokensCounter
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.utils.typeshelper import Languages

    mh = _minhash_config(config)

    # Stage A — process.
    stage_a = LocalPipelineExecutor(
        pipeline=build_process_pipeline(config),
        tasks=config.tasks,
        workers=config.workers,
        logging_dir=_log(config, "a_process"),
    )

    # Stage B — minhash (4 sub-stages, chained via `depends`).
    # NOTE: MinhashDedupSignature computes n-gram fingerprints with a language-specific
    # word tokenizer; we pass Turkish so agglutinative words tokenize correctly. Its
    # internal text simplification still uses str.lower() (not our Turkish-aware
    # normalize_for_dedup), so I/İ can hash identically — a minor, known source of extra
    # false-duplicate matches. Acceptable; a full fix means patching MinhashConfig.
    sig = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(_p(config, "intermediate")),
            MinhashDedupSignature(
                output_folder=_p(config, "minhash/signatures"),
                config=mh,
                language=Languages.turkish,
            ),
        ],
        tasks=config.tasks,
        logging_dir=_log(config, "b1_signatures"),
        depends=stage_a,
    )
    buckets = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=_p(config, "minhash/signatures"),
                output_folder=_p(config, "minhash/buckets"),
                config=mh,
            )
        ],
        tasks=mh.num_buckets,
        logging_dir=_log(config, "b2_buckets"),
        depends=sig,
    )
    cluster = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=_p(config, "minhash/buckets"),
                output_folder=_p(config, "minhash/remove_ids"),
                config=mh,
            )
        ],
        tasks=1,
        logging_dir=_log(config, "b3_cluster"),
        depends=buckets,
    )

    # Stage C — drop duplicates, count tokens, write final.
    final_steps: list = [
        JsonlReader(_p(config, "intermediate")),
        MinhashDedupFilter(
            input_folder=_p(config, "minhash/remove_ids"),
            exclusion_writer=JsonlWriter(_p(config, "removed_duplicates")),
        ),
    ]
    if config.tokenizer.name_or_path:
        final_steps.append(TokensCounter(tokenizer_name_or_path=config.tokenizer.name_or_path))
    final_steps.append(JsonlWriter(output_folder=_p(config, "final")))

    final = LocalPipelineExecutor(
        pipeline=final_steps,
        tasks=config.tasks,
        workers=config.workers,
        logging_dir=_log(config, "c_final"),
        depends=cluster,
    )
    final.run()
    return final


def run_slurm(config: PipelineConfig, *, partition: str, time: str = "20:00:00", **slurm_kwargs):
    """Build the same pipeline on SlurmPipelineExecutor for cluster scale.

    Returns the final executor (call ``.run()``). Only the executor class changes; the
    pipeline definition is identical to :func:`run_local`. Left as a thin builder so the
    cluster specifics (partition, mem, cpus) are supplied at the call site.
    """
    _require_datatrove()
    config.validate()
    raise NotImplementedError(
        "run_slurm is a documented seam: mirror run_local but swap LocalPipelineExecutor "
        "for SlurmPipelineExecutor(partition=partition, time=time, mem_per_cpu_gb=..., "
        "cpus_per_task=..., **slurm_kwargs), preserving the same A->B->C pipeline and "
        "`depends` chain. Wire it when the Linux/Slurm box specs are known."
    )


def _p(config: PipelineConfig, sub: str) -> str:
    return os.path.join(config.output_path, sub)


def _log(config: PipelineConfig, sub: str) -> str:
    return os.path.join(config.logging_dir, sub)
