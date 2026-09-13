from pathlib import Path

import pytest

from rag.chunking import chunk_page
from rag.models import PDFPage
from rag.vector_store import VectorStore


def test_chunk_keeps_source_page_and_text() -> None:
    page = PDFPage(source="law.pdf", page=3, text="alpha beta gamma")
    chunks = chunk_page(page, chunk_size=10, chunk_overlap=2)

    assert chunks
    assert chunks[0].source == "law.pdf"
    assert chunks[0].page == 3
    assert chunks[0].chunk_id == "law.pdf:p3:c0"
    assert chunks[0].text


def test_missing_vector_database_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Vector database not found"):
        VectorStore.load(tmp_path / "vector_db", model_name="test-model")
