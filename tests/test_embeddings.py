from __future__ import annotations

import numpy as np

from rag.embeddings import Embedder


def test_local_vector_backend_is_deterministic_without_model_download() -> None:
    embedder = Embedder("local:char-ngram-v1")
    first = embedder.encode(["食品過敏原標示規定", "低脂營養宣稱"])
    second = embedder.encode(["食品過敏原標示規定", "低脂營養宣稱"])

    assert first.shape == (2, 4096)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
