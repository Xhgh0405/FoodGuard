"""Public search API for the FoodGuard RAG index."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

from .config import vector_store_dir
from .vector_store import VectorStore


@lru_cache(maxsize=4)
def _load_store(directory: str) -> VectorStore:
    """Load one vector store once per server process.

    MCP may call several tools during one conversation. Reusing the loaded
    embedder avoids reloading the sentence-transformer model for every call.
    """

    return VectorStore.load(Path(directory))


def search(query: str, top_k: int = 5, vector_db: Path | str | None = None) -> list[dict[str, Any]]:
    """Search the local FAISS index and return score plus source metadata."""

    directory = Path(vector_db) if vector_db is not None else vector_store_dir()
    store = _load_store(str(directory.resolve()))
    return store.search(query, top_k=top_k)
