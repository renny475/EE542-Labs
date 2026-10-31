"""Turn a manifest source's file into plain text, page by page.

``Page`` is ``(page_number, text)``; the page number is 1-based for PDFs and
``None`` for HTML and for the single-page result of a custom extractor.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from manifest import Source

Page = tuple[int | None, str]

_DROP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")
_LINE_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "tr")
_CFR_F_START = re.compile(r"^\(f\) Students in colleges", re.MULTILINE)
_CFR_G_START = re.compile(r"^\(g\) ", re.MULTILINE)


class ExtractError(ValueError):
    """The file could not be reduced to the text the manifest promised."""


def pdf_pages(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    return [(i, (page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]


def _line_text(tag: Tag) -> str:
    if tag.name == "tr":
        cells = [c.get_text(" ", strip=True) for c in tag.find_all(("td", "th"), recursive=False)]
        return " | ".join(c for c in cells if c)
    return tag.get_text(" ", strip=True)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup
    lines: list[str] = []
    for tag in root.find_all(_LINE_TAGS):
        # A <p> inside an <li> would otherwise be emitted twice.
        if tag.find_parent(_LINE_TAGS) is not None and tag.name != "tr":
            continue
        text = _line_text(tag)
        if text:
            lines.append(text)
    return re.sub(r"\n{2,}", "\n", "\n".join(lines)).strip()


def _normalise_digits(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def strip_repeated_lines(pages: list[Page], min_fraction: float = 0.3) -> list[Page]:
    """Drop lines that recur on most pages (print headers/footers, URLs, page counters)."""
    if len(pages) < 3:
        return pages
    pages_with_line: dict[str, int] = {}
    for _, text in pages:
        for key in {_normalise_digits(line) for line in text.splitlines() if line.strip()}:
            pages_with_line[key] = pages_with_line.get(key, 0) + 1
    threshold = max(2.0, min_fraction * len(pages))  # a line on one page is never "repeated"
    repeated = {key for key, count in pages_with_line.items() if count >= threshold}
    return [
        (n, "\n".join(line for line in text.splitlines() if _normalise_digits(line) not in repeated))
        for n, text in pages
    ]


def cfr_paragraph_f(text: str) -> str:
    """Slice 8 CFR 214.2 down to paragraph (f).

    The govinfo PDF opens with a table of contents that also lists the
    paragraph headings, so the body is the *last* ``(f)`` heading and the
    first ``(g)`` heading after it.
    """
    starts = list(_CFR_F_START.finditer(text))
    if not starts:
        raise ExtractError("8 CFR 214.2: '(f) Students in colleges' heading not found")
    start = starts[-1].start()
    end_match = _CFR_G_START.search(text, starts[-1].end())
    if end_match is None:
        raise ExtractError("8 CFR 214.2: '(g)' heading after paragraph (f) not found")
    return text[start:end_match.start()].rstrip()


def extract_source(source: Source, data_dir: Path) -> list[Page]:
    path = source.path(data_dir)
    if source.kind == "pdf":
        pages = strip_repeated_lines(pdf_pages(path))
    elif source.kind == "html":
        pages = [(None, html_to_text(path.read_text(encoding="utf-8", errors="replace")))]
    else:
        raise ExtractError(f"{source.id}: unsupported kind {source.kind!r}")
    if source.extractor == "cfr_paragraph_f":
        return [(None, cfr_paragraph_f("\n".join(text for _, text in pages)))]
    if source.extractor is not None:
        raise ExtractError(f"{source.id}: unknown extractor {source.extractor!r}")
    return pages
