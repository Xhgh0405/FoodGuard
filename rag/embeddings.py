"""Sentence-transformers embedding adapter."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. Install dependencies with "
                "`py -3.11 -m pip install -r requirements.txt`."
            ) from exc
        self.model_name = model_name
        offline = os.getenv("FOODGUARD_OFFLINE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.model = SentenceTransformer(model_name, local_files_only=offline)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype="float32")
        vectors = self.model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
