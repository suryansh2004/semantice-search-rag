import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .embedder import SentenceTransformerEmbedder
from .schemas import Chunk, Document
from .text_splitter import split_text
from .vector_store import VectorStore


def load_documents(input_file: str | Path) -> list[Document]:
    documents: list[Document] = []
    with Path(input_file).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                documents.append(Document.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL document at line {line_number}: {exc}") from exc
    return documents


def build_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        text_chunks = split_text(doc.text, chunk_size=chunk_size, overlap=chunk_overlap)
        for index, chunk_text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.id}::chunk-{index:04d}",
                    doc_id=doc.id,
                    title=doc.title,
                    text=chunk_text,
                    source=doc.source,
                )
            )
    return chunks


def ingest(input_file: str, index_dir: str, model_name: str, chunk_size: int, chunk_overlap: int) -> None:
    documents = load_documents(input_file)
    chunks = build_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were created. Check your input data.")

    embedder = SentenceTransformerEmbedder(model_name)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    store = VectorStore(embeddings=embeddings, chunks=chunks)
    store.save(index_dir)

    metadata = {
        "model_name": model_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "backend": store.backend,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_at": datetime.now(UTC).isoformat(),
    }
    Path(index_dir, "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Build semantic search index")
    parser.add_argument("--input", required=True, help="Path to JSONL documents")
    parser.add_argument("--index-dir", default=config.index_dir, help="Directory to save index artifacts")
    parser.add_argument("--model-name", default=config.embedding_model, help="Sentence Transformer model name")
    parser.add_argument("--chunk-size", type=int, default=config.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=config.chunk_overlap)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(
        input_file=args.input,
        index_dir=args.index_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
