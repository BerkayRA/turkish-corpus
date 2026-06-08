"""Tests for the shared source-ingestion contract (pure)."""

import gzip
import json

import pytest

from turkish_corpus.sources.base import SourceInfo, make_record, write_records

WIKI = SourceInfo(name="wikipedia", license="CC BY-SA 3.0", register="encyclopedic")


class TestMakeRecord:
    def test_shape_and_provenance(self):
        rec = make_record("Merhaba dünya.", "wiki-1", WIKI, url="http://x", year=2024)
        assert rec["id"] == "wiki-1"
        assert rec["text"] == "Merhaba dünya."
        assert rec["metadata"]["source"] == "wikipedia"
        assert rec["metadata"]["license"] == "CC BY-SA 3.0"
        assert rec["metadata"]["register"] == "encyclopedic"
        assert rec["metadata"]["url"] == "http://x"
        assert rec["metadata"]["year"] == 2024


class TestWriteRecords:
    def test_writes_gzip_jsonl_and_counts(self, tmp_path):
        records = [make_record(f"belge {i} içeriği", f"d{i}", WIKI) for i in range(3)]
        n = write_records(records, str(tmp_path / "wikipedia"))
        assert n == 3
        path = tmp_path / "wikipedia" / "00000.jsonl.gz"
        assert path.is_file()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 3
        assert rows[0]["metadata"]["source"] == "wikipedia"
        # Turkish characters stored literally, not \u-escaped.
        raw = path.read_bytes()
        assert "içeriği".encode() in gzip.decompress(raw)

    def test_skips_empty_and_short_text(self, tmp_path):
        records = [
            make_record("", "empty", WIKI),
            make_record("   ", "blank", WIKI),
            make_record("yeterince uzun bir metin", "ok", WIKI),
        ]
        n = write_records(records, str(tmp_path / "s"), min_chars=5)
        assert n == 1

    def test_rejects_record_missing_keys(self, tmp_path):
        with pytest.raises(ValueError, match="id/text"):
            write_records([{"text": "x but no id", "metadata": {}}], str(tmp_path / "s"))

    def test_rejects_malformed_even_when_short(self, tmp_path):
        # Structure is validated BEFORE the length skip, so a malformed record whose text is
        # also below min_chars still raises instead of being silently dropped.
        with pytest.raises(ValueError, match="id/text"):
            write_records([{"text": "x", "metadata": {}}], str(tmp_path / "s"), min_chars=5)

    def test_plain_jsonl_when_not_gz(self, tmp_path):
        n = write_records(
            [make_record("metin", "d1", WIKI)], str(tmp_path / "s"), shard_name="part.jsonl"
        )
        assert n == 1
        assert (tmp_path / "s" / "part.jsonl").is_file()
