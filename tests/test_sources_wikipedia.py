"""Tests for the Turkish Wikipedia ingester.

The record-building helper :func:`_records` is pure (no datasets/network), so it is tested
directly against fake article dicts. :func:`ingest_wikipedia` is tested with ``load_dataset``
monkeypatched to a small in-memory iterable, so the full write path is exercised without any
Hub access — keeping these in the ``sources`` marker only because they live alongside the
optional-extra source ingesters.
"""

import gzip
import json

import pytest

from turkish_corpus.sources.wikipedia import WIKIPEDIA, _records, ingest_wikipedia

pytestmark = pytest.mark.sources


def _fake_articles() -> list[dict]:
    """A few rows shaped like wikimedia/wikipedia, including a too-short one to be skipped."""
    return [
        {
            "id": "12",
            "title": "Türkiye",
            "url": "https://tr.wikipedia.org/wiki/T%C3%BCrkiye",
            "text": "Türkiye, Avrupa ve Asya kıtalarında yer alan bir ülkedir.",
        },
        {
            "id": "34",
            "title": "Ankara",
            "url": "https://tr.wikipedia.org/wiki/Ankara",
            "text": "Ankara, Türkiye'nin başkenti ve ikinci büyük şehridir.",
        },
        {"id": "99", "title": "Boş", "url": "https://tr.wikipedia.org/wiki/Bo%C5%9F", "text": ""},
    ]


class TestRecords:
    def test_yields_correctly_shaped_records(self):
        recs = list(_records(_fake_articles()))
        assert len(recs) == 3  # _records does not filter; write_records does
        first = recs[0]
        assert first["id"] == "wiki-12"
        assert first["text"].startswith("Türkiye")
        meta = first["metadata"]
        assert meta["source"] == "wikipedia"
        assert meta["license"] == WIKIPEDIA.license
        assert meta["register"] == "encyclopedic"
        assert meta["title"] == "Türkiye"
        assert meta["url"].endswith("T%C3%BCrkiye")

    def test_limit_caps_iteration(self):
        recs = list(_records(_fake_articles(), limit=2))
        assert [r["id"] for r in recs] == ["wiki-12", "wiki-34"]

    def test_nonpositive_limit_means_all(self):
        assert len(list(_records(_fake_articles(), limit=-1))) == 3
        assert len(list(_records(_fake_articles(), limit=0))) == 3


class TestIngestWikipedia:
    def test_writes_gzip_jsonl_and_skips_empty(self, tmp_path, monkeypatch):
        def fake_load_dataset(name, dump, split, streaming):
            assert name == "wikimedia/wikipedia"
            assert dump == "20231101.tr"
            assert split == "train" and streaming is True
            return iter(_fake_articles())

        # datasets is imported lazily inside ingest_wikipedia; swap the module so the
        # `from datasets import load_dataset` line binds our fake instead.
        import sys
        import types

        fake_mod = types.ModuleType("datasets")
        fake_mod.load_dataset = fake_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_mod)

        out = tmp_path / "wikipedia"
        written = ingest_wikipedia(str(out), dump="20231101.tr")

        # The empty article is dropped by write_records' min_chars guard.
        assert written == 2
        path = out / "00000.jsonl.gz"
        assert path.is_file()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert [r["id"] for r in rows] == ["wiki-12", "wiki-34"]
        assert rows[0]["metadata"]["register"] == "encyclopedic"

    def test_limit_passed_through(self, tmp_path, monkeypatch):
        import sys
        import types

        fake_mod = types.ModuleType("datasets")
        fake_mod.load_dataset = lambda *a, **k: iter(_fake_articles())
        monkeypatch.setitem(sys.modules, "datasets", fake_mod)

        written = ingest_wikipedia(str(tmp_path / "w"), limit=1)
        assert written == 1
