import pytest
from fpdf import FPDF

from documents import paragraphs, read_upload


def make_pdf(text: str | None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    if text is None:
        pdf.rect(20, 20, 50, 50, style="F")
    else:
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def test_read_upload_txt_and_pdf():
    assert read_upload("Take aspirin daily.".encode(), "notes.txt") == "Take aspirin daily."
    assert read_upload(b"# Heading\nbody", "a.MD") == "# Heading\nbody"
    assert "Take aspirin daily." in read_upload(make_pdf("Take aspirin daily."), "d.pdf")


def test_read_upload_scanned_pdf_error():
    with pytest.raises(ValueError) as exc:
        read_upload(make_pdf(None), "scan.pdf")
    assert str(exc.value) == "No readable text found. Scanned PDFs need OCR first."


def test_read_upload_rejects_docx():
    with pytest.raises(ValueError):
        read_upload(b"x", "a.docx")


def test_read_upload_empty_txt_error():
    with pytest.raises(ValueError, match="No readable text found"):
        read_upload(b"  \n ", "a.txt")


def test_paragraphs_splits_blank_lines_and_long_blocks():
    assert paragraphs("one\n\n\ntwo\nstill two\n\n  \nthree") == ["one", "two\nstill two", "three"]
    long_block = "\n".join(["x" * 100] * 8)
    parts = paragraphs("short\n\n" + long_block)
    assert parts[0] == "short"
    assert parts[1:] == ["x" * 100] * 8
    assert paragraphs("") == []
