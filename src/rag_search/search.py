import argparse
import json

from .config import AppConfig
from .embedder import SentenceTransformerEmbedder
from .schemas import SearchResult
from .vector_store import VectorStore


class SemanticSearchEngine:
    def __init__(self, index_dir: str, model_name: str) -> None:
        self.store = VectorStore.load(index_dir)
        self.embedder = SentenceTransformerEmbedder(model_name)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self.embedder.encode([query])[0]
        return self.store.search(query_embedding=query_embedding, top_k=top_k)


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Search semantic index")
    parser.add_argument("--index-dir", default=config.index_dir)
    parser.add_argument("--model-name", default=config.embedding_model)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=config.default_top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SemanticSearchEngine(index_dir=args.index_dir, model_name=args.model_name)
    results = engine.search(query=args.query, top_k=args.top_k)
    print(json.dumps([result.model_dump() for result in results], indent=2))


if __name__ == "__main__":
    main()
