"""PDF ingestion and section chunking for the retrieval component."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pymupdf


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    heading: str
    text: str


def read_pdf(path: Path) -> list[dict]:
    pages = []
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDF: provide an unlocked demo copy.")
        for number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            if not text:
                raise ValueError(f"Page {number} has no extractable text; OCR is required.")
            pages.append({"page": number, "text": text})
    if not pages:
        raise ValueError("The PDF has no pages.")
    return pages


def make_chunks(pages: list[dict], max_chars: int = 4500) -> list[Chunk]:
    """Keep colon-headed sections intact, including medication qualifiers.

    Oversized sections fail explicitly instead of splitting away conditions.
    PDFs without recognizable headings use a whole page as one section.
    """
    chunks = []
    for page in pages:
        heading, lines = "Page context", []

        def emit():
            text = "\n".join(lines).strip()
            if not text:
                return
            if len(text) > max_chars:
                raise ValueError(
                    f"Page {page['page']}, section {heading!r} exceeds {max_chars} "
                    "characters. Review and split the section before processing."
                )
            chunks.append(Chunk(f"c{len(chunks) + 1:04d}", page["page"], heading, text))

        for line in page["text"].splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) <= 100 and line.endswith(":"):
                emit()
                heading, lines = line[:-1], [line]
            else:
                lines.append(line)
        emit()
    return chunks
