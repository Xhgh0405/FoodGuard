"""Document discovery and PDF text extraction using PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .models import PDFPage


def iter_pdf_pages(documents_path: Path) -> Iterator[PDFPage]:
    """Yield non-empty pages from every supported source below ``documents_path``.

    ``source`` is a path relative to the documents directory so the value is
    stable across machines and can be displayed as a citation in the UI.
    PDF page numbers are one-based. Text and Markdown sources are treated as
    page 1 so official web-based source notes can also be searched.
    """

    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required to read PDFs. Install dependencies with "
            "`py -3.11 -m pip install -r requirements.txt`."
        ) from exc

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_path}. "
            "Create it and place TFDA PDF files inside."
        )
    if not documents_path.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {documents_path}")

    source_paths = sorted(
        path
        for path in documents_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    if not source_paths:
        raise FileNotFoundError(
            f"No PDF, Markdown, or text files found in {documents_path}. "
            "Place official source files there before building the index."
        )

    for source_path in source_paths:
        source = source_path.relative_to(documents_path).as_posix()
        if source_path.suffix.lower() in {".md", ".txt"}:
            try:
                text = source_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                raise RuntimeError(f"Unable to read text source: {source_path}. {exc}") from exc
            if text:
                yield PDFPage(source=source, page=1, text=text)
            continue
        try:
            with fitz.open(source_path) as document:
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        yield PDFPage(source=source, page=page_number, text=text)
        except Exception as exc:
            raise RuntimeError(f"Unable to read PDF: {source_path}. {exc}") from exc
