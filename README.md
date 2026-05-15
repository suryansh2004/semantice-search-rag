# Semantic Search and RAG Service

A production-style semantic search and retrieval augmented generation starter project. It builds an embedding index from JSONL documents, evaluates retrieval quality, and serves search plus grounded answer endpoints through FastAPI.

## What It Demonstrates

- JSONL document ingestion and validation
- Configurable text chunking
- Sentence Transformer embeddings
- FAISS vector search with a NumPy fallback
- Retrieval metrics: Recall@K and MRR@K
- FastAPI serving with health, readiness, metadata, search, and answer routes
- Docker and Docker Compose deployment
- Render blueprint for cloud deployment
- A realistic internal knowledge-base sample corpus and labeled eval set

## Project Structure

```text
semantic-search-rag/
  data/
    sample_docs.jsonl
    eval_queries.jsonl
  deploy/
    render.yaml
  docs/
    deployment.md
  src/rag_search/
    api.py
    config.py
    embedder.py
    evaluate.py
    ingest.py
    rag.py
    schemas.py
    search.py
    text_splitter.py
    vector_store.py
  tests/
  Dockerfile
  docker-compose.yml
  Makefile
  pyproject.toml
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e ".[dev]"
```

## Configuration

Copy the example environment file when you want to override defaults:

```bash
cp .env.example .env
```

Supported variables:

```text
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_INDEX_DIR=artifacts/index
RAG_CHUNK_SIZE=450
RAG_CHUNK_OVERLAP=80
RAG_DEFAULT_TOP_K=5
```

If deployment cannot access Hugging Face, set `RAG_EMBEDDING_MODEL` to a local model directory that was already downloaded.

## Run The Pipeline

Build the index:

```bash
python -m src.rag_search.ingest \
  --input data/sample_docs.jsonl \
  --index-dir artifacts/index
```

Search:

```bash
python -m src.rag_search.search \
  --index-dir artifacts/index \
  --query "How should a team deploy a RAG service?" \
  --top-k 3
```

Evaluate retrieval:

```bash
python -m src.rag_search.evaluate \
  --index-dir artifacts/index \
  --eval-file data/eval_queries.jsonl \
  --top-k 3
```

## Run The API

```bash
uvicorn src.rag_search.api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /health    lightweight process health
GET  /ready     verifies index artifacts exist
GET  /metadata  returns index/backend/model details
POST /search    semantic retrieval
POST /answer    simple grounded answer with sources
```

Example:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What should I monitor in a search service?", "top_k":3}'
```

## Docker

```bash
docker compose up --build
```

The image builds the sample index during Docker build. For larger real corpora, build the index as a separate job and restore artifacts from persistent storage.

## Deployment

See [docs/deployment.md](docs/deployment.md). A Render blueprint is included at [deploy/render.yaml](deploy/render.yaml).

## Data Format

Documents are JSONL:

```json
{"id":"kb-001","title":"Document title","text":"Full document text...","source":"source-name"}
```

Evaluation queries are JSONL:

```json
{"query":"What is semantic search?","relevant_ids":["kb-001"]}
```

## Tests

```bash
pytest -q
```          
