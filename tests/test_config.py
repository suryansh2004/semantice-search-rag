import pytest

from src.rag_search.config import AppConfig


def test_app_config_reads_environment_at_instantiation(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", "custom/index")
    monkeypatch.setenv("RAG_DEFAULT_TOP_K", "7")

    config = AppConfig()

    assert config.index_dir == "custom/index"
    assert config.default_top_k == 7


def test_app_config_rejects_invalid_integer(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-a-number")

    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE must be an integer"):
        AppConfig()
