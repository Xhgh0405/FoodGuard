"""Data models used by the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A searchable passage and the source location it came from."""

    source: str
    page: int
    chunk_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDFPage:
    source: str
    page: int
    text: str
