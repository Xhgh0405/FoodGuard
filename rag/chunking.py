"""Text chunking that keeps each passage tied to its PDF page."""

from __future__ import annotations

from .models import Chunk, PDFPage


def chunk_page(page: PDFPage, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    """Split one PDF page into overlapping character-based chunks.

    The values control retrieval context size only; they are not legal or
    nutritional thresholds.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    text = " ".join(page.text.split())
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    start = 0
    chunk_number = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    source=page.source,
                    page=page.page,
                    chunk_id=f"{page.source}:p{page.page}:c{chunk_number}",
                    text=chunk_text,
                )
            )
            chunk_number += 1
        if end == len(text):
            break
        start += step
    return chunks


def chunk_pages(
    pages: list[PDFPage], chunk_size: int = 1200, chunk_overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks
