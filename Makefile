.PHONY: install install-dev ingest search evaluate api test docker docker-run

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

ingest:
	python -m src.rag_search.ingest --input data/sample_docs.jsonl --index-dir artifacts/index

search:
	python -m src.rag_search.search --index-dir artifacts/index --query "How should a team deploy a RAG service?" --top-k 3

evaluate:
	python -m src.rag_search.evaluate --index-dir artifacts/index --eval-file data/eval_queries.jsonl --top-k 3

api:
	uvicorn src.rag_search.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

docker:
	docker build -t semantic-search-rag .

docker-run:
	docker compose up --build
