"""The shared ingestion contract every corpus source implements.

An *ingester* is any callable that yields :func:`make_record` dicts; :func:`write_records`
persists them as gzipped, datatrove ``JsonlReader``-shaped JSONL (``text_key="text"``,
``id_key="id"``), so the existing pipeline cleans every source identically:

    raw JSONL (this module's shape) --> tc-run-hplt --source jsonl --> cleaned JSONL

Records always carry provenance in ``metadata`` (source name, license, register) so the
final blend manifest can report token counts per source and license — important for the
research/permissive licensing posture and for reproducibility.

Pure stdlib (json, gzip, os) so it imports and tests anywhere, without the ``sources`` or
``academic`` extras.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["SourceInfo", "make_record", "write_records", "REGISTERS"]

# The register a source contributes — used for blend weighting and reporting. Not an enum
# so new registers can be added freely; these are the expected values.
REGISTERS = (
    "encyclopedic",  # Wikipedia
    "legal",         # mevzuat, Resmî Gazete, courts, TBMM
    "academic",      # YÖKTEZ theses, DergiPark
    "news",          # TS/BOUN news corpora
    "web",           # HPLT / Common Crawl
    "conversational",  # OpenSubtitles
)


@dataclass(frozen=True)
class SourceInfo:
    """Provenance for one source, stamped into every record's metadata.

    Parameters
    ----------
    name:
        Short id, also used as the output subdirectory (e.g. ``"wikipedia"``).
    license:
        Human-readable license/terms (e.g. ``"CC BY-SA 3.0"``, ``"public (official text)"``,
        ``"research"``). Carried through so the blend manifest can audit licensing.
    register:
        One of :data:`REGISTERS` (free-form, but prefer these).
    description:
        Optional one-line note on the source.
    """

    name: str
    license: str
    register: str
    description: str = ""


def make_record(text: str, doc_id: str, source: SourceInfo, **metadata) -> dict:
    """Build one datatrove-ready record with stamped provenance.

    Returns ``{"id": doc_id, "text": text, "metadata": {source/license/register, **extra}}``.
    Extra keyword args (url, title, year, …) ride in ``metadata`` for provenance.
    """
    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "source": source.name,
            "license": source.license,
            "register": source.register,
            **metadata,
        },
    }


def write_records(
    records: Iterable[dict],
    out_dir: str,
    *,
    shard_name: str = "00000.jsonl.gz",
    min_chars: int = 1,
) -> int:
    """Write ``records`` to ``out_dir/shard_name`` as gzipped JSONL; return the count written.

    Records with empty/whitespace-only or ``< min_chars`` text are skipped (the downstream
    quality filters would drop them anyway; skipping here keeps the raw shard lean). The
    output directory is created if needed. UTF-8, ``ensure_ascii=False`` so Turkish
    characters are stored literally rather than escaped.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, shard_name)
    written = 0
    opener = gzip.open if shard_name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            text = rec.get("text") or ""
            if len(text.strip()) < min_chars:
                continue
            if "id" not in rec or "text" not in rec:
                raise ValueError(f"record missing required id/text keys: {sorted(rec)}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written
