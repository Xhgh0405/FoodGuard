"""Task-specific retrieval, relevance filtering, and de-duplication."""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Any, Callable

from rag import search as default_search


SearchFunction = Callable[[str, int], list[dict[str, Any]]]


def _minimum_score() -> float:
    try:
        return float(os.getenv("RAG_MIN_SCORE", "0.35"))
    except ValueError:
        return 0.35


def retrieve_evidence(
    query: str,
    *,
    source_hints: tuple[str, ...] = (),
    top_k: int = 3,
    search_function: SearchFunction | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only relevant, unique evidence for one analysis task."""

    searcher = search_function or default_search
    retrieved = searcher(query, max(10, top_k * 3))
    min_score = _minimum_score()
    scored = [
        item
        for item in retrieved
        if float(item.get("score", 0.0)) >= min_score
        and item.get("source")
        and item.get("text")
    ]

    if source_hints:
        # A task must not silently consume a high-scoring chunk from another
        # regulation. If no hinted source survives the score filter, the
        # caller receives insufficient evidence instead of an unrelated hit.
        scored = [
            item
            for item in scored
            if any(
                hint.casefold() in f"{item.get('source', '')} {item.get('text', '')}".casefold()
                for hint in source_hints
            )
        ]

    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    unique: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, int, str]] = set()
    for item in scored:
        identity = (
            str(item.get("source", "")),
            int(item.get("page", 0)),
            str(item.get("chunk_id", "")),
        )
        if identity in seen_ids:
            continue
        normalized = " ".join(str(item.get("text", "")).split())
        if any(
            SequenceMatcher(None, normalized[:1000], " ".join(str(existing.get("text", "")).split())[:1000]).ratio()
            >= 0.93
            for existing in unique
        ):
            continue
        seen_ids.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break

    debug = {
        "query": query,
        "retrieved_count": len(retrieved),
        "retrieved_chunks": [
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
            }
            for item in retrieved
        ],
        "after_relevance_count": len(scored),
        "returned_count": len(unique),
        "minimum_score": min_score,
        "source_hints": list(source_hints),
    }
    return unique, debug
