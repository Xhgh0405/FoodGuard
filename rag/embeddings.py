"""Sentence-transformers embedding adapter."""

from __future__ import annotations

import json
import hashlib
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Iterable

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._ollama_model: str | None = None
        self._ollama_base_url = ""
        self._local_hash = model_name.startswith("local:")
        self._local_dimensions = 4096
        if self._local_hash:
            return
        if model_name.startswith("ollama:"):
            self._ollama_model = model_name.split(":", 1)[1].strip()
            if not self._ollama_model:
                raise ValueError("Ollama embedding model name must not be empty.")
            self._ollama_base_url = os.getenv(
                "OLLAMA_EMBEDDING_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/")
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. Install dependencies with "
                "`py -3.11 -m pip install -r requirements.txt`."
            ) from exc
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
        if self._local_hash:
            return self._encode_local(values)
        if self._ollama_model:
            return self._encode_ollama(values)
        vectors = self.model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def _encode_local(self, values: list[str]) -> np.ndarray:
        """Create deterministic dense char n-gram vectors without downloads.

        This is a lightweight vector-retrieval backend for deployments where
        a transformer model cannot be downloaded.  It is intentionally not
        described as semantic embedding: the n-gram representation improves
        vector indexing and multilingual matching while remaining fully
        reproducible on Streamlit Cloud and local machines.
        """

        vectors = np.zeros((len(values), self._local_dimensions), dtype="float32")
        for row, value in enumerate(values):
            text = " ".join(str(value).casefold().split())
            if not text:
                continue
            grams: list[str] = [text]
            for size in (2, 3, 4, 5):
                grams.extend(text[index : index + size] for index in range(len(text) - size + 1))
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % self._local_dimensions
                vectors[row, bucket] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _encode_ollama(self, values: list[str]) -> np.ndarray:
        """Encode a batch through Ollama's native /api/embed endpoint."""

        # Avoid sending thousands of chunks in one request.  Ollama accepts a
        # list input, but a bounded batch keeps model memory and request time
        # predictable during a full corpus rebuild.
        batch_size = int(os.getenv("OLLAMA_EMBEDDING_BATCH_SIZE", "16"))
        if batch_size <= 0:
            raise ValueError("OLLAMA_EMBEDDING_BATCH_SIZE must be greater than zero.")
        all_embeddings: list[list[float]] = []
        timeout = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "120"))
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            payload = json.dumps(
                {"model": self._ollama_model, "input": batch}, ensure_ascii=False
            ).encode("utf-8")
            request = Request(
                f"{self._ollama_base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Ollama embedding request failed at {self._ollama_base_url}: {exc}"
                ) from exc
            embeddings = result.get("embeddings") if isinstance(result, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise RuntimeError("Ollama embedding response returned an unexpected shape.")
            all_embeddings.extend(embeddings)
        vectors = np.asarray(all_embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(values):
            raise RuntimeError("Ollama embedding response returned an unexpected shape.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Ollama returned a zero-length embedding.")
        return vectors / norms
