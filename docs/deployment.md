# Deployment

This project ships with a Docker image that builds the retrieval index during image creation and serves the FastAPI app with Uvicorn.

## Local Docker

```bash
docker compose up --build
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How should a team deploy a RAG service?", "top_k":3}'
```

## Render

1. Push the repository to GitHub.
2. Create a new Blueprint on Render and select `deploy/render.yaml`.
3. Keep `/ready` as the health check path.
4. Watch the first build carefully because the Docker image downloads the embedding model.

## Production Notes

- Use `RAG_EMBEDDING_MODEL` to point at a local model directory if outbound downloads are blocked.
- Keep `artifacts/` out of git for large real indexes; build the index during Docker build or restore it from object storage.
- For larger datasets, replace the in-process FAISS/NumPy store with Qdrant, Chroma, Milvus, or managed vector search.
- Run evaluation after each data refresh and record Recall@K/MRR@K before promoting a new index.
