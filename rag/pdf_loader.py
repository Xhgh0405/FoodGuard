"""Document discovery and PDF text extraction using PyMuPDF."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from .models import PDFPage


class _VisibleTextParser(HTMLParser):
    """Extract visible text from an HTML source without adding a dependency."""

    _ignored_tags = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _html_to_text(source_path: Path) -> str:
    parser = _VisibleTextParser()
    parser.feed(source_path.read_text(encoding="utf-8", errors="ignore"))
    return " ".join(parser.parts).strip()


def iter_pdf_pages(documents_path: Path) -> Iterator[PDFPage]:
    """Yield non-empty pages from every supported source below ``documents_path``.

    ``source`` is a path relative to the documents directory so the value is
    stable across machines and can be displayed as a citation in the UI.
    PDF page numbers are one-based. Text, Markdown, and HTML sources are
    treated as page 1 so official web-based source notes can also be searched.
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
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt", ".html", ".htm"}
    )
    if not source_paths:
        raise FileNotFoundError(
            f"No PDF, Markdown, or text files found in {documents_path}. "
            "Place official source files there before building the index."
        )

    for source_path in source_paths:
        source = source_path.relative_to(documents_path).as_posix()
        if source_path.suffix.lower() in {".md", ".txt", ".html", ".htm"}:
            try:
                text = (
                    _html_to_text(source_path)
                    if source_path.suffix.lower() in {".html", ".htm"}
                    else source_path.read_text(encoding="utf-8").strip()
                )
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
