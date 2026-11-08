import pytest
from fpdf import FPDF

from documents import MAX_PARAGRAPH, paragraphs, read_upload


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


def test_paragraphs_splits_blank_lines_and_joins_wrapped_lines():
    assert paragraphs("one\n\n\ntwo\nstill two\n\n  \nthree") == ["one", "two still two", "three"]
    assert paragraphs("") == []
    assert paragraphs("  \n\n ") == []


def squash(text):
    return " ".join(text.split())


# What pypdf gives for a one-page AVS: visual lines, no blank lines anywhere.
SHORT_PDF_TEXT = (
    "Discharge Instructions\n"
    "Take amoxicillin 500 mg by mouth three times daily for 7 days. Finish all of\n"
    "the pills even if you feel better.\n"
    "Call your doctor if your fever is above 101 F or\n"
    "you have a new rash.\n"
    "Diet:\n"
    "- Drink plenty of fluids\n"
    "- Avoid alcohol while taking this\n"
    "medicine\n"
    "Patient: Sam Lee (fictional)\n"
    "Follow-up: see Dr. Lee\n"
    "in 2 weeks.\n"
    "1. Bring this sheet\n"
    "2) Bring your pill bottles"
)


def test_paragraphs_short_pdf_text_keeps_units_and_joins_wraps():
    assert paragraphs(SHORT_PDF_TEXT) == [
        "Discharge Instructions",
        "Take amoxicillin 500 mg by mouth three times daily for 7 days. Finish all of "
        "the pills even if you feel better.",
        "Call your doctor if your fever is above 101 F or you have a new rash.",
        "Diet:",
        "- Drink plenty of fluids",
        "- Avoid alcohol while taking this medicine",
        "Patient: Sam Lee (fictional)",
        "Follow-up: see Dr. Lee in 2 weeks.",
        "1. Bring this sheet",
        "2) Bring your pill bottles",
    ]


def test_paragraphs_abbreviation_at_line_end_does_not_split():
    assert paragraphs("Call Dr.\nPriya Raman at the clinic if the pain is not getting better.") == [
        "Call Dr. Priya Raman at the clinic if the pain is not getting better."
    ]


LONG_SENTENCES = [
    f"Sentence {n} says to take the medicine number {n} with a full glass of water at noon."
    for n in range(1, 21)
]


def wrap(text, width=70):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]


def test_paragraphs_long_wrapped_text_splits_at_sentences_under_the_cap():
    text = " ".join(LONG_SENTENCES)
    parts = paragraphs("\n".join(wrap(text)))
    assert len(parts) > 1
    assert all(len(p) <= MAX_PARAGRAPH for p in parts)
    assert " ".join(parts) == text
    for p in parts:  # every part is whole sentences
        assert p.startswith("Sentence ") and p.endswith("at noon.")


def test_paragraphs_single_overlong_sentence_is_split_by_words():
    text = " ".join(["word"] * 400) + "."
    parts = paragraphs(text)
    assert all(len(p) <= MAX_PARAGRAPH for p in parts)
    assert " ".join(parts) == text


def test_paragraphs_long_block_with_blank_lines_also_capped():
    block = " ".join(LONG_SENTENCES)
    parts = paragraphs("Short heading\n\n" + block)
    assert parts[0] == "Short heading"
    assert all(len(p) <= MAX_PARAGRAPH for p in parts)


def wrapped_pdf(text: str) -> bytes:
    """A real PDF whose text wraps across visual lines, with no blank lines."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(120, 6, text)
    return bytes(pdf.output())


def test_wrapped_pdf_short_page_gives_sentence_units():
    text = (
        "Take amoxicillin 500 mg by mouth three times daily for 7 days and finish every pill. "
        "Call your doctor if your fever is above 101 F or you have a new rash on your skin."
    )
    raw = read_upload(wrapped_pdf(text), "d.pdf")
    assert raw.count("\n") >= 2  # pypdf really returns wrapped visual lines
    # The wrapped lines join back up; the sentence boundary falls mid-line, so
    # the two sentences stay one unit (well under the cap).
    assert paragraphs(raw) == [text]


def test_wrapped_pdf_long_page_never_splits_mid_sentence():
    raw = read_upload(wrapped_pdf(" ".join(LONG_SENTENCES)), "d.pdf")
    parts = [squash(p) for p in paragraphs(raw)]
    assert all(len(p) <= MAX_PARAGRAPH for p in parts)
    assert " ".join(parts) == " ".join(LONG_SENTENCES)
    for p in parts:
        assert p.startswith("Sentence ") and p.endswith("at noon.")


def test_read_upload_normalizes_ligatures():
    assert read_upload("Con\ufb01rm the \ufb02uid plan.".encode(), "a.txt") == "Confirm the fluid plan."


def test_read_upload_malformed_pdf_any_error_is_a_clean_value_error(monkeypatch):
    import documents

    class Broken:
        def __init__(self, stream):
            raise TypeError("unexpected structure")

    monkeypatch.setattr(documents, "PdfReader", Broken)
    with pytest.raises(ValueError, match="The PDF could not be read."):
        read_upload(b"%PDF-1.4 junk", "d.pdf")
