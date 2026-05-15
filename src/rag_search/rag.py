from .schemas import SearchResult


def generate_grounded_answer(query: str, results: list[SearchResult]) -> str:
    if not results:
        return "I could not find enough relevant context to answer the question."

    best = results[0]
    source_titles = ", ".join(dict.fromkeys(result.title for result in results))

    return (
        f"Based on the retrieved documents, the most relevant context is from '{best.title}'. "
        f"{best.text} Sources considered: {source_titles}."
    )
