import os
from dataclasses import dataclass, field


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class AppConfig:
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    chunk_size: int = field(default_factory=lambda: _get_int("RAG_CHUNK_SIZE", 450))
    chunk_overlap: int = field(default_factory=lambda: _get_int("RAG_CHUNK_OVERLAP", 80))
    default_top_k: int = field(default_factory=lambda: _get_int("RAG_DEFAULT_TOP_K", 5))
    index_dir: str = field(default_factory=lambda: os.getenv("RAG_INDEX_DIR", "artifacts/index"))
