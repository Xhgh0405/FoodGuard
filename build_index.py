"""Build the FoodGuard FAISS index from official sources under documents/."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.chunking import chunk_pages
from rag.config import documents_dir, embedding_model_name, vector_store_dir
from rag.embeddings import Embedder
from rag.pdf_loader import iter_pdf_pages
from rag.vector_store import VectorStore


def build_index(
    documents_path: Path | None = None,
    output_path: Path | None = None,
    model_name: str | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> int:
    source_dir = documents_path or documents_dir()
    destination = output_path or vector_store_dir()
    embedding_name = model_name or embedding_model_name()

    pages = list(iter_pdf_pages(source_dir))
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = VectorStore(destination, Embedder(embedding_name))
    store.build(chunks)
    store.save()
    print(f"Indexed {len(pages)} source pages into {len(chunks)} chunks.")
    print(f"Vector database: {destination}")
    print(f"Embedding model: {embedding_name}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        help="Official source directory; default: documents/",
    )
    parser.add_argument("--output", type=Path, default=None, help="FAISS directory; default: data/vector_store/")
    parser.add_argument("--model", default=None, help="sentence-transformers model name")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    try:
        build_index(
            documents_path=args.documents,
            output_path=args.output,
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (FileNotFoundError, NotADirectoryError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
