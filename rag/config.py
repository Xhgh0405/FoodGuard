"""Configuration helpers for the local FoodGuard RAG index."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def documents_dir() -> Path:
    _load_env()
    return _project_path(os.getenv("TFDA_DOCUMENTS_DIR", "documents"), "documents")


def vector_store_dir() -> Path:
    _load_env()
    return _project_path(os.getenv("VECTOR_STORE_DIR", "data/vector_store"), "data/vector_store")


def embedding_model_name() -> str:
    _load_env()
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
