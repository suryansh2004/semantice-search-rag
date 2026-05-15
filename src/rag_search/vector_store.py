import json
import pickle
from pathlib import Path

import numpy as np

from .schemas import Chunk, SearchResult


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) == 0:
            raise ValueError("at least one chunk is required")

        self.chunks = chunks
        self.backend = "numpy"
        self.faiss_index = None

        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.faiss_index.add(self.embeddings)
            self.backend = "faiss"
        except Exception:
            self.faiss_index = None

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"query embedding dimension {query.shape[1]} does not match "
                f"index dimension {self.embeddings.shape[1]}"
            )

        top_k = min(top_k, len(self.chunks))

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query, top_k)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            scores = np.dot(self.embeddings, query[0])
            top_indices = np.argsort(scores)[::-1][:top_k]
            pairs = [(int(i), float(scores[i])) for i in top_indices]

        results: list[SearchResult] = []
        for idx, score in pairs:
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    score=float(score),
                )
            )
        return results

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "embeddings.pkl").open("wb") as f:
            pickle.dump(self.embeddings, f)

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

        metadata = {
            "backend": self.backend,
            "embedding_dim": int(self.embeddings.shape[1]),
            "num_chunks": len(self.chunks),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        path = Path(index_dir)
        missing = [
            filename
            for filename in ("embeddings.pkl", "chunks.jsonl")
            if not (path / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing index artifacts in {path}: {', '.join(missing)}")

        with (path / "embeddings.pkl").open("rb") as f:
            embeddings = pickle.load(f)

        chunks: list[Chunk] = []
        with (path / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))

        return cls(embeddings=embeddings, chunks=chunks)
