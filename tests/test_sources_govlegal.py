"""Tests for the government/legal ingesters — no network (datasets is monkeypatched)."""

import gzip
import json

import pytest

from turkish_corpus.sources import govlegal

pytestmark = pytest.mark.sources


class TestRecordFromRow:
    def test_uses_text_key(self):
        rec = govlegal._record_from_row({"id": "L1", "text": "Madde 1 — metin."}, 0)
        assert rec["id"] == "mevzuat-L1"
        assert rec["text"] == "Madde 1 — metin."
        assert rec["metadata"]["register"] == "legal"
        assert rec["metadata"]["source"] == "mevzuat"
        assert rec["metadata"]["license"] == "public (official text)"

    def test_falls_back_to_content_key(self):
        rec = govlegal._record_from_row({"id": 7, "content": "kanun metni"}, 0)
        assert rec["text"] == "kanun metni"
        assert rec["id"] == "mevzuat-7"

    def test_falls_back_to_madde_key(self):
        rec = govlegal._record_from_row({"madde": "Madde içeriği"}, 5)
        assert rec["text"] == "Madde içeriği"
        # No id in the row → fall back to the streaming index.
        assert rec["id"] == "mevzuat-5"

    def test_returns_none_when_textless(self):
        assert govlegal._record_from_row({"id": "x", "title": "başlık"}, 0) is None
        assert govlegal._record_from_row({"text": ""}, 0) is None


class TestRecordsGenerator:
    def test_skips_textless_and_respects_limit(self):
        rows = [
            {"text": "bir"},
            {"title": "no text here"},  # skipped, must not consume budget
            {"content": "iki"},
            {"madde": "üç"},
        ]
        out = list(govlegal._records(rows, limit=2))
        assert [r["text"] for r in out] == ["bir", "iki"]

    def test_negative_limit_yields_all(self):
        rows = [{"text": f"m{i}"} for i in range(4)]
        assert len(list(govlegal._records(rows, limit=-1))) == 4


class TestScaffolds:
    @pytest.mark.parametrize(
        "func",
        [
            govlegal.ingest_resmi_gazete,
            govlegal.ingest_court_decisions,
            govlegal.ingest_tbmm_tutanak,
        ],
    )
    def test_scaffold_raises_with_helpful_message(self, func, tmp_path):
        with pytest.raises(NotImplementedError) as exc:
            func(str(tmp_path))
        message = str(exc.value)
        assert "scaffold" in message.lower()
        # The message names the polite-crawler approach so the next implementer is oriented.
        assert "make_record" in message


class TestIngestMevzuat:
    def test_writes_gzip_jsonl_without_network(self, tmp_path, monkeypatch):
        fake_rows = [
            {"id": "A", "text": "Birinci madde metni"},
            {"title": "no usable text"},  # skipped
            {"content": "İkinci madde içeriği"},
        ]

        def fake_load_dataset(name, split, streaming):
            assert streaming is True  # must stream, never fully download
            return iter(fake_rows)

        # Patch the symbol where govlegal imports it (lazy `from datasets import ...`).
        import datasets

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

        out_dir = tmp_path / "mevzuat"
        written = govlegal.ingest_mevzuat(str(out_dir), limit=-1)
        assert written == 2

        path = out_dir / "00000.jsonl.gz"
        assert path.is_file()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert [r["text"] for r in rows] == ["Birinci madde metni", "İkinci madde içeriği"]
        assert all(r["metadata"]["register"] == "legal" for r in rows)
        # Turkish characters stored literally, not \\u-escaped.
        assert "İkinci".encode() in gzip.decompress(path.read_bytes())
