FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RAG_INDEX_DIR=/app/artifacts/index

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m src.rag_search.ingest --input data/sample_docs.jsonl --index-dir artifacts/index

EXPOSE 8000

CMD ["uvicorn", "src.rag_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
