"""Tests for the academic ingesters — no network, no real PDFs (extraction is stubbed)."""

from pathlib import Path

import pytest

pytest.importorskip("pypdf")  # academic extra; skip cleanly if it's not installed

from turkish_corpus.sources import academic  # noqa: E402

pytestmark = pytest.mark.academic


class TestRecordsFromPdfs:
    def test_only_extractable_pdfs_become_records(self, tmp_path):
        # Three PDFs; the stub extractor "succeeds" on two and returns None (scanned) on one.
        born_digital = {tmp_path / "a.pdf", tmp_path / "c.pdf"}

        def stub_extractor(path: Path) -> str | None:
            return f"text of {path.stem}" if path in born_digital else None

        paths = [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"]
        records = academic._records_from_pdfs(
            paths, academic.DERGIPARK, stub_extractor, limit=-1
        )
        out = list(records)

        # Only the born-digital PDFs are emitted, with academic provenance + stable doc ids.
        assert [r["id"] for r in out] == ["dergipark-a", "dergipark-c"]
        assert all(r["metadata"]["register"] == "academic" for r in out)
        assert all(r["metadata"]["source"] == "dergipark" for r in out)
        assert [r["text"] for r in out] == ["text of a", "text of c"]
        # The filename rides in metadata for provenance.
        assert out[0]["metadata"]["filename"] == "a.pdf"
        # The scanned PDF (b.pdf) is tallied as needing OCR.
        assert records.needs_ocr == 1

    def test_limit_counts_only_emitted_records(self, tmp_path):
        # A scanned PDF in the middle must not consume the emit budget.
        def stub_extractor(path: Path) -> str | None:
            return None if path.stem == "skip" else f"text {path.stem}"

        paths = [
            tmp_path / "one.pdf",
            tmp_path / "skip.pdf",
            tmp_path / "two.pdf",
            tmp_path / "three.pdf",
        ]
        records = academic._records_from_pdfs(
            paths, academic.YOKTEZ, stub_extractor, limit=2
        )
        out = list(records)
        assert [r["id"] for r in out] == ["yoktez-one", "yoktez-two"]
        # Iteration stopped at the 2-record cap before reaching three.pdf, but it had already
        # passed the scanned skip.pdf, which is counted.
        assert records.needs_ocr == 1

    def test_extractor_exception_is_treated_as_needs_ocr(self, tmp_path):
        # A blow-up in the extractor must not crash the batch; treat it as unreadable.
        def boom_extractor(path: Path) -> str | None:
            raise ValueError("corrupt PDF")

        records = academic._records_from_pdfs(
            [tmp_path / "x.pdf"], academic.DERGIPARK, boom_extractor, limit=-1
        )
        assert list(records) == []
        assert records.needs_ocr == 1


class TestExtractPdfText:
    def test_returns_none_for_scanned_pdf(self, tmp_path, monkeypatch):
        # Simulate a scanned PDF: pypdf opens it but every page yields empty text, and the
        # pdfplumber fallback is also empty → extract_pdf_text returns None (needs OCR).
        class _EmptyPage:
            def extract_text(self) -> str:
                return ""

        class _FakeReader:
            def __init__(self, *_args, **_kwargs) -> None:
                self.pages = [_EmptyPage(), _EmptyPage()]

        import pypdf

        monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)
        # Force the pdfplumber fallback to contribute nothing either.
        monkeypatch.setattr(academic, "_extract_with_pdfplumber", lambda _path: "")

        fake_pdf = tmp_path / "scanned.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")  # never actually parsed (reader is faked)

        assert academic.extract_pdf_text(fake_pdf) is None

    def test_returns_text_for_born_digital_pdf(self, tmp_path, monkeypatch):
        # pypdf yields plenty of text → returned without touching pdfplumber.
        long_text = "Tez metni. " * 50  # well over the min_chars threshold

        class _Page:
            def extract_text(self) -> str:
                return long_text

        class _FakeReader:
            def __init__(self, *_args, **_kwargs) -> None:
                self.pages = [_Page()]

        import pypdf

        monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)

        fake_pdf = tmp_path / "thesis.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        result = academic.extract_pdf_text(fake_pdf)
        assert result is not None
        assert "Tez metni." in result


class TestDownloadScaffolds:
    @pytest.mark.parametrize(
        "func",
        [academic.download_dergipark, academic.download_yoktez],
    )
    def test_scaffold_raises_with_helpful_message(self, func, tmp_path):
        with pytest.raises(NotImplementedError) as exc:
            func(str(tmp_path))
        message = str(exc.value).lower()
        assert "scaffold" in message
        # The message orients the next implementer on download + the polite-fetch approach.
        assert "download" in message
