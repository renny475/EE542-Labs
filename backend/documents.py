import io
import re
from pathlib import PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_PARAGRAPH = 600
NO_TEXT_ERROR = "No readable text found. Scanned PDFs need OCR first."


def read_upload(data: bytes, filename: str) -> str:
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PyPdfError, ValueError, KeyError) as exc:
            raise ValueError("The PDF could not be read.") from exc
    elif suffix in {".txt", ".md"}:
        try:
            pages = [data.decode("utf-8-sig")]
        except UnicodeDecodeError as exc:
            raise ValueError("The text file must be UTF-8.") from exc
    else:
        raise ValueError("Use a PDF, TXT, or Markdown file.")
    if not any(page.strip() for page in pages):
        raise ValueError(NO_TEXT_ERROR)
    return "\n\n".join(page.strip("\n") for page in pages if page.strip())


def paragraphs(text: str) -> list[str]:
    result: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) > MAX_PARAGRAPH:
            result.extend(line.strip() for line in block.splitlines() if line.strip())
        else:
            result.append(block)
    return result
