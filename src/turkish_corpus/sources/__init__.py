"""Register-diverse corpus sources (roadmap step 2).

Each *source* (Turkish Wikipedia, government/legal text, academic theses, …) is ingested
into **raw** datatrove-ready JSONL via a small shared contract (:mod:`.base`), then cleaned
by the EXISTING pipeline (``tc-run-hplt --source jsonl``), then mixed by
:mod:`turkish_corpus.blend` into the final register-diverse corpus. Ingesters only produce
``{"id","text","metadata"}`` records — all normalization, filtering, KVKK PII scrubbing,
and dedup are reused from steps 1-2 (the DRY hand-off), not reimplemented per source.

Only :mod:`.base` is imported here (pure). Individual ingester modules import their heavy
deps (``datasets``, ``pypdf``, …) lazily and live behind the ``sources`` / ``academic``
extras.
"""

from __future__ import annotations

from .base import SourceInfo, make_record, write_records

__all__ = ["SourceInfo", "make_record", "write_records"]
