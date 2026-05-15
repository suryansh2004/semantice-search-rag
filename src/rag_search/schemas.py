from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    text: str
    source: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    source: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
