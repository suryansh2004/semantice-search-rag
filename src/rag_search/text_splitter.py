def split_text(text: str, chunk_size: int = 450, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        candidate = normalized[start:end]

        if end < len(normalized):
            sentence_end = max(candidate.rfind("."), candidate.rfind("?"), candidate.rfind("!"))
            if sentence_end >= chunk_size // 2:
                end = start + sentence_end + 1
                candidate = normalized[start:end]

        chunks.append(candidate.strip())

        if end == len(normalized):
            break
        start = max(0, end - overlap)

    return chunks
