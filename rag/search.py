"""Public search API for the FoodGuard RAG index."""

from __future__ import annotations

import re
from pathlib import Path
from functools import lru_cache
from typing import Any

from .chunking import chunk_pages
from .config import documents_dir, vector_store_dir
from .pdf_loader import iter_pdf_pages
from .vector_store import VectorStore


@lru_cache(maxsize=4)
def _load_store(directory: str) -> VectorStore:
    """Load one vector store once per server process.

    MCP may call several tools during one conversation. Reusing the loaded
    embedder avoids reloading the sentence-transformer model for every call.
    """

    return VectorStore.load(Path(directory))


@lru_cache(maxsize=2)
def _load_keyword_chunks(directory: str) -> tuple[dict[str, Any], ...]:
    """Load small searchable passages for use when FAISS is unavailable."""

    pages = list(iter_pdf_pages(Path(directory)))
    return tuple(chunk.to_dict() for chunk in chunk_pages(pages))


def _query_terms(query: str) -> list[str]:
    normalized = query.casefold()
    terms: list[str] = []
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        # Chinese has no word separators. Keep the full phrase and short
        # n-grams so「糖尿病可以喝嗎」still matches「糖尿病」in a document.
        terms.append(sequence)
        for size in (2, 3, 4):
            terms.extend(sequence[index : index + size] for index in range(len(sequence) - size + 1))
    terms.extend(re.findall(r"[a-z0-9][a-z0-9_-]+", normalized))
    return list(dict.fromkeys(term for term in terms if len(term) >= 2))


def _keyword_search(query: str, top_k: int) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return []

    matches: list[dict[str, Any]] = []
    for item in _load_keyword_chunks(str(documents_dir().resolve())):
        haystack = f"{item.get('source', '')} {item.get('text', '')}".casefold()
        hit_count = sum(1 for term in terms if term in haystack)
        if not hit_count:
            continue
        # This score is deliberately conservative: it only lets an explicit
        # keyword hit pass the normal evidence threshold; it is not a legal
        # confidence score.
        result = dict(item)
        result["score"] = min(0.89, 0.35 + 0.08 * hit_count)
        matches.append(result)
    matches.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return matches[:top_k]


def search(query: str, top_k: int = 5, vector_db: Path | str | None = None) -> list[dict[str, Any]]:
    """Search the local FAISS index and return score plus source metadata."""

    directory = Path(vector_db) if vector_db is not None else vector_store_dir()
    try:
        results = _load_store(str(directory.resolve())).search(query, top_k=top_k)
    except (FileNotFoundError, ImportError, RuntimeError):
        return _keyword_search(query, top_k)

    # A valid index can still return only weak semantic matches for a short
    # Chinese question such as「糖尿病可以喝嗎」. Add explicit document hits
    # so known sources are not discarded merely because the embedding score is
    # low.
    keyword_results = _keyword_search(query, top_k)
    combined = results + keyword_results
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for item in sorted(combined, key=lambda value: float(value.get("score", 0.0)), reverse=True):
        identity = (str(item.get("source", "")), int(item.get("page", 0)), str(item.get("chunk_id", "")))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique
