import pytest

from src.rag_search.text_splitter import split_text


def test_split_text_returns_chunks():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = split_text(text, chunk_size=20, overlap=5)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, overlap=10)
