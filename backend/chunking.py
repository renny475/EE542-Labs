"""Split extracted pages into overlapping chunks that remember their page."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract import Page


@dataclass
class ChunkText:
    index: int
    page: int | None
    content: str


def chunk_pages(pages: list[Page], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[ChunkText]:
    kept = [(n, text) for n, text in pages if text.strip()]
    if not kept:
        return []

    # Join the pages and remember where each one starts so a chunk can be
    # attributed to the page containing its first character.
    starts: list[int] = []
    page_numbers: list[int | None] = []
    parts: list[str] = []
    offset = 0
    for n, text in kept:
        starts.append(offset)
        page_numbers.append(n)
        parts.append(text)
        offset += len(text) + 1  # the "\n" separator
    joined = "\n".join(parts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[ChunkText] = []
    cursor = 0
    for piece in splitter.split_text(joined):
        if not piece.strip():
            continue
        found = joined.find(piece, cursor)
        if found == -1:  # the splitter stripped whitespace; fall back to the cursor
            found = cursor
        cursor = found
        page = page_numbers[bisect.bisect_right(starts, found) - 1]
        chunks.append(ChunkText(index=len(chunks), page=page, content=piece))
    return chunks
