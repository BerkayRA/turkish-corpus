"""Turkish Wikipedia (Vikipedi) source ingester (roadmap step 2, encyclopedic register).

Streams the ``wikimedia/wikipedia`` dataset from the Hugging Face Hub and emits raw
datatrove-ready JSONL via the shared contract (:mod:`.base`); the EXISTING pipeline cleans
it (``tc-run-hplt --source jsonl``), so no normalization/filtering happens here.

Why ``wikimedia/wikipedia`` (not the raw ``*wiki-latest-pages-articles.xml`` dumps): it
ships **pre-parsed clean prose** — wikitext markup, templates, tables, and infoboxes are
already stripped — so we get article body text without bundling a wikitext parser. The
``dump`` argument is a date-coded config id, e.g. ``"20231101.tr"`` (snapshot date + wiki
language code). Available configs change as Wikimedia publishes new snapshots; verify the
current set on the dataset card / config dropdown at
https://huggingface.co/datasets/wikimedia/wikipedia before pinning a value, because an
unknown ``dump`` raises at ``load_dataset`` time.

License note: Wikipedia text is CC BY-SA 3.0/4.0 — **share-alike**. Any corpus or model
release that redistributes this text inherits the attribution + share-alike obligations,
which is why the license is stamped into every record's metadata for the blend manifest's
licensing audit.

``datasets`` is imported lazily inside :func:`ingest_wikipedia` so this module (and the
record-building helper :func:`_records`) imports and unit-tests without the ``sources`` extra
or any network access.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import SourceInfo, make_record, write_records

__all__ = ["WIKIPEDIA", "ingest_wikipedia"]

WIKIPEDIA = SourceInfo(
    name="wikipedia",
    license="CC BY-SA 3.0/4.0",
    register="encyclopedic",
    description="Turkish Wikipedia (Vikipedi) articles",
)


def _records(dataset: Iterable[dict], limit: int = -1) -> Iterator[dict]:
    """Yield :func:`make_record` dicts from raw ``wikimedia/wikipedia`` article rows.

    Pulled out of :func:`ingest_wikipedia` so the record shaping (id/title/url provenance,
    ``limit`` cut-off) is unit-testable against a plain list of dicts — no Hub or network.
    Each article carries its title and source URL into ``metadata`` for provenance; the id
    is namespaced ``wiki-<id>`` so it stays unique once blended with other sources. A
    ``limit > 0`` stops iteration early (cheap smoke runs); ``limit <= 0`` means all.
    """
    for i, article in enumerate(dataset):
        if 0 < limit <= i:
            return
        yield make_record(
            text=article["text"],
            doc_id=f"wiki-{article['id']}",
            source=WIKIPEDIA,
            title=article.get("title"),
            url=article.get("url"),
        )


def ingest_wikipedia(out_dir: str, *, dump: str = "20231101.tr", limit: int = -1) -> int:
    """Stream Turkish Wikipedia into raw JSONL under ``out_dir``; return records written.

    Loads ``wikimedia/wikipedia`` config ``dump`` (a date-coded id like ``"20231101.tr"`` —
    see module docstring on verifying available configs) in **streaming** mode so we never
    materialize the full dataset in memory, shapes each article via :func:`_records`, and
    persists with :func:`write_records` (which skips empty/short rows). ``limit > 0`` caps
    the article count for smoke runs; ``limit <= 0`` ingests everything.
    """
    # Lazy import: keeps this module importable without the `sources` extra (datasets).
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset("wikimedia/wikipedia", dump, split="train", streaming=True)
    return write_records(_records(dataset, limit), out_dir)
