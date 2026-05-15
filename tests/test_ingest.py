from src.rag_search.ingest import build_chunks, load_documents
from src.rag_search.schemas import Document


def test_load_documents_reads_realistic_sample_data():
    documents = load_documents("data/sample_docs.jsonl")

    assert len(documents) >= 20
    assert documents[0].id.startswith("kb-")


def test_build_chunks_preserves_document_metadata():
    document = Document(
        id="kb-test",
        title="Metadata",
        text="Sentence one. Sentence two explains metadata.",
        source="unit-test",
    )

    chunks = build_chunks([document], chunk_size=80, chunk_overlap=10)

    assert chunks[0].doc_id == "kb-test"
    assert chunks[0].title == "Metadata"
    assert chunks[0].source == "unit-test"
