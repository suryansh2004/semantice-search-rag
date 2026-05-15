import argparse
import json
from pathlib import Path

from .config import AppConfig
from .search import SemanticSearchEngine


def load_eval_queries(eval_file: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(eval_file).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" not in row or "relevant_ids" not in row:
                raise ValueError(f"Line {line_number} must include query and relevant_ids")
            rows.append(row)
    return rows


def evaluate(index_dir: str, model_name: str, eval_file: str, top_k: int) -> dict[str, float]:
    engine = SemanticSearchEngine(index_dir=index_dir, model_name=model_name)
    eval_queries = load_eval_queries(eval_file)

    hits = 0
    reciprocal_rank_sum = 0.0

    for row in eval_queries:
        relevant_ids = set(row["relevant_ids"])
        results = engine.search(row["query"], top_k=top_k)
        retrieved_doc_ids = [result.doc_id for result in results]

        first_relevant_rank = None
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / first_relevant_rank

    total = len(eval_queries)
    return {
        f"recall@{top_k}": hits / total if total else 0.0,
        f"mrr@{top_k}": reciprocal_rank_sum / total if total else 0.0,
        "queries": float(total),
    }


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Evaluate semantic search retrieval")
    parser.add_argument("--index-dir", default=config.index_dir)
    parser.add_argument("--model-name", default=config.embedding_model)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--top-k", type=int, default=config.default_top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        index_dir=args.index_dir,
        model_name=args.model_name,
        eval_file=args.eval_file,
        top_k=args.top_k,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
