from functools import lru_cache
import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException

from .config import AppConfig
from .rag import generate_grounded_answer
from .schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse
from .search import SemanticSearchEngine

app = FastAPI(title="Semantic Search and RAG API", version="1.0.0")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> SemanticSearchEngine:
    config = AppConfig()
    return SemanticSearchEngine(index_dir=config.index_dir, model_name=config.embedding_model)


def get_config() -> AppConfig:
    return AppConfig()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    config = get_config()
    index_path = Path(config.index_dir)
    required = ["embeddings.pkl", "chunks.jsonl"]
    missing = [name for name in required if not (index_path / name).exists()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Index is not ready. Missing artifacts: {', '.join(missing)}",
        )
    return {"status": "ready", "index_dir": config.index_dir}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    config = get_config()
    try:
        engine = get_engine()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "embedding_model": config.embedding_model,
        "index_dir": config.index_dir,
        "backend": engine.store.backend,
        "chunks": len(engine.store.chunks),
        "embedding_dim": int(engine.store.embeddings.shape[1]),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    started_at = perf_counter()
    try:
        results = get_engine().search(query=request.query, top_k=request.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info(
        "search query_length=%s top_k=%s results=%s latency_ms=%.2f",
        len(request.query),
        request.top_k,
        len(results),
        (perf_counter() - started_at) * 1000,
    )
    return SearchResponse(query=request.query, results=results)


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    started_at = perf_counter()
    try:
        results = get_engine().search(query=request.query, top_k=request.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer_text = generate_grounded_answer(query=request.query, results=results)
    logger.info(
        "answer query_length=%s top_k=%s sources=%s latency_ms=%.2f",
        len(request.query),
        request.top_k,
        len(results),
        (perf_counter() - started_at) * 1000,
    )
    return AnswerResponse(query=request.query, answer=answer_text, sources=results)
