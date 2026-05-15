import numpy as np
import pytest

from src.rag_search.schemas import Chunk
from src.rag_search.vector_store import VectorStore


def make_chunk(chunk_id: str = "doc-1::chunk-0000") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        title="Test Document",
        text="This is a test chunk.",
        source="test",
    )


def test_vector_store_rejects_empty_index():
    with pytest.raises(ValueError, match="at least one chunk"):
        VectorStore(embeddings=np.empty((0, 3), dtype=np.float32), chunks=[])


def test_vector_store_clamps_top_k_to_available_chunks():
    store = VectorStore(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunks=[make_chunk("doc-1::chunk-0000"), make_chunk("doc-2::chunk-0000")],
    )

    results = store.search(query_embedding=np.array([1.0, 0.0], dtype=np.float32), top_k=10)

    assert len(results) == 2
    assert results[0].chunk_id == "doc-1::chunk-0000"


def test_vector_store_rejects_wrong_query_dimension():
    store = VectorStore(
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        chunks=[make_chunk()],
    )

    with pytest.raises(ValueError, match="does not match index dimension"):
        store.search(query_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
