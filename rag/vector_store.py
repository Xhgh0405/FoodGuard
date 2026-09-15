"""FAISS persistence and similarity search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import Embedder
from .models import Chunk


INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "config.json"


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for the vector database. Install dependencies with "
            "`py -3.11 -m pip install -r requirements.txt`."
        ) from exc
    return faiss


class VectorStore:
    def __init__(self, directory: Path, embedder: Embedder) -> None:
        self.directory = directory
        self.embedder = embedder
        self.index: Any | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No text chunks were created; check that the PDFs contain extractable text.")
        vectors = self.embedder.encode(chunk.text for chunk in chunks)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RuntimeError("Embedding generation returned an unexpected shape.")

        faiss = _import_faiss()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = [chunk.to_dict() for chunk in chunks]

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("The vector index has not been built.")
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss = _import_faiss()
        faiss.write_index(self.index, str(self.directory / INDEX_FILENAME))
        (self.directory / METADATA_FILENAME).write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.directory / CONFIG_FILENAME).write_text(
            json.dumps({"embedding_model": self.embedder.model_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, model_name: str | None = None) -> "VectorStore":
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not directory.exists() or not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {directory}. "
                "Run `py build_index.py` after placing PDF files in documents/."
            )

        faiss = _import_faiss()
        saved_model_name = model_name
        config_path = directory / CONFIG_FILENAME
        if saved_model_name is None and config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_model_name = config.get("embedding_model")
        if not saved_model_name:
            raise RuntimeError(
                f"Embedding model configuration is missing from vector database: {directory}"
            )
        store = cls(directory, Embedder(saved_model_name))
        store.index = faiss.read_index(str(index_path))
        store.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if store.index.ntotal != len(store.metadata):
            raise RuntimeError("FAISS index and metadata record counts do not match.")
        return store

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.index is None:
            raise RuntimeError("The vector index is not loaded.")

        query_vector = self.embedder.encode([query])
        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, limit)
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            result = dict(self.metadata[index])
            result["score"] = float(score)
            results.append(result)
        return results
